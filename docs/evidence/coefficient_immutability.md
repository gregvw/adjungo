# C-15.7 coefficient-immutability evidence

Status: **supporting evidence and failure history**. The current requirements
and supported boundary remain in [NUMERICS.md C-15.7](../../NUMERICS.md#c-15-7).
This document preserves the reasoning and observations extracted from that
clause in the worktree based on `5c19cc8`, after the replacement-provider
witness and its clause clarification were added. It creates no new policy.

The narrative records successive corrections, including superseded intermediate
rules. Read those passages as the history of a failure, not as additional
current requirements. In particular, an early reachability argument counted a
view's owner; the later **Storage is not reconstruction** passage withdraws
that argument. The operative rule and the full caller boundary are in the
[current clause](../../NUMERICS.md#c-15-7-envelope).

The [95-defect campaign](#historical-campaign) is historical evidence at its
**822-test baseline**, with its original failure counts. Moving this record,
adding the later witness, or running a larger current suite does not rerun or
refresh that campaign.

- [Scalar explicit-Euler fixture and closed-form displacement](#scalar-euler-fixture)
- [Storage traversal withdrawal](#storage-is-not-reconstruction)
- [Current supported/excluded boundary](../../NUMERICS.md#c-15-7-envelope)
- [Historical injection campaign](#historical-campaign)
- [Executable witnesses](../../tests/test_coefficient_immutability.py)

## Failure history

`AffineDynamics.__init__` copies its coefficient arrays and sets
`writeable=False`. That is what makes it safe for `F` to return `self._M`
itself rather than a copy: the tape the backward sweeps read holds references
to that one buffer, so a buffer that could change between the forward solve
and the adjoint would seed the adjoint with a matrix the forward solve never
used.

**The invariant is that the coefficient data cannot be written, and it must
hold for every instance the caller can obtain.** Two things were assumed to
establish it and did not.

### The deep copy

`copy.deepcopy` did not preserve the flag. NumPy defines
`ndarray.__deepcopy__`, and it returns a writeable array, so the copy arrived
with the ownership guarantee already gone. `pickle` goes through `__reduce__`,
which records the flag, and `copy.copy` shares the original buffers, so the
deep copy was alone in this — which is why it survived M6 and M7 unnoticed.

The cure is `__setstate__`, the one hook that both the default copy and the
default pickle protocol run, re-establishing the constructor's postcondition
for every name in `adjungo/core/affine.py::_COEFFICIENT_BUFFERS`.
Reconstructing through `__deepcopy__` instead would have discarded state a
subclass had added, which is a worse failure for a smaller gain — and that
choice obliges `__setstate__` to accept the `(instance_dict, slot_values)`
form, since a subclass carrying `__slots__` overrides no guaranteed member and
so is verified.

The postcondition is *owning* as well as frozen. A frozen view over a
writeable base is not immutable data, and there were two: `pickle`
reconstructs an `ndarray` as a view over the pickled `bytes`, and
`np.array(b, copy=True).reshape(-1)` returned a view whose `base` stayed
writeable, so an explicitly supplied `b` could be changed through `p.b.base`
while every flag on `p.b` read read-only.

The copy protocol is therefore part of the guarantee, and
[C-16.9](../../NUMERICS.md#c-16)'s comparison is extended to `__setstate__`, `__getstate__`,
`__deepcopy__`, `__copy__`, `__replace__`, `__reduce__` and `__reduce_ex__`. A
subclass overriding `__setstate__` alone leaves every other comparison in
`affine_dynamics_verified` passing, so without that the cure is one method
away from being undone. This covers the *default* protocols: a `copyreg`
reducer registered externally for the class runs ahead of them, and is C-1
territory.

### Construction

The second assumption was that anything reaching `__init__` is safe. It is
not. A subclass writing

```python
super().__init__(M, C, b)
self._M = self._M.copy()
```

looks defensive and is the reverse: NumPy returns a **writeable** copy. Such a
subclass overrides no guaranteed member and no copy hook, so every structural
comparison passes and it is certified affine with mutable coefficients.

`affine_dynamics_verified` therefore also checks the buffers themselves, and
**raises `MutableCoefficients` rather than returning `False`**. That asymmetry
is the substance of this clause. Every other refusal in that function is safe
to answer quietly, because the general route then computes the same derivative
more slowly. This one is not.

### Refusing is a guard; only immutability is a cure

C-Q7 posed re-freezing and a writeability check as alternatives. They are not:
one is the cure and the other is a guard against the invariant being false for
some reason the cure does not cover, and both are needed. A refusal alone would
not repair anything, because the general route retains what `F` returns in the
same way the affine route does.

<a id="scalar-euler-fixture"></a>

This is not an argument, it is a measurement. Fixture: `n = ν = 1`, `M = 0.4`,
`b = 0`, `y₀ = 0.8`, `t_span = (0, 0.5)`, `N = 2` so `h = 0.25`,
`u = (0.3, 0.2)`, `explicit_euler`, `J = ½y₂²`. A coefficient is overwritten
with `M' = 0.9` on the first callback of the backward sweep, the earliest
moment at which the two sweeps can disagree. With `a = 1 + hM`, the gradient is
`∂J/∂u₀ = hCy₂a` and `∂J/∂u₁ = hCy₂`, so replacing `a` by `a'` in the adjoint
alone displaces the first component by `hCy₂(a' − a) = h²Cy₂(M' − M)` —
**one** factor of `C`, not two:

| Quantity | `C = 1` | `C = 2` |
|---|---|---|
| `y₁`, `y₂` | `0.955`, `1.1005` | `1.03`, `1.233` |
| `∂J/∂u` exact | `(0.3026375, 0.275125)` | `(0.678150, 0.61650)` |
| `∂J/∂u` with the overwrite | `(0.337028125, 0.275125)` | `(0.7552125, 0.61650)` |
| Displacement `h²Cy₂(M' − M)` | `0.034390625` | `0.0770625` |

The displacement is derived in closed form rather than read off a broken run,
per [C-14.1](../../NUMERICS.md#c-14). The `C = 2` column is carried because at `C = 1` a
spurious square is invisible, and an earlier draft of this clause had one.
`∂J/∂u₁` never multiplies by the state Jacobian and is untouched, which
localises the defect to the one propagation step that reads the overwritten
matrix.

After the cure the overwrite raises `ValueError` at the write itself, which is
the [C-7](../../NUMERICS.md#c-7) direction.

**"Every route" would be too strong and is not claimed.** The monolithic
reference path in `adjungo/validation/reference.py` assembles its own arrays
and copies values out of `F`, and is *not* displaced by this mutation. What
decides the matter is not whether a route reads `F` but whether it retains the
object `F` returned, which the `GLMOptimizer` stage-solver paths do.

### A check that raises may not sit behind checks that return

The guard above was correct, and unreachable by exactly the problems with the
strongest claim on it. It was written as the last of four checks in
`affine_dynamics_verified`, after three that answer quietly, so a problem
refused for any unrelated reason never reached it — and a quiet refusal is
exactly the outcome the preceding section establishes as unsafe. Problems that
were otherwise eligible did reach it, and every test of the guard exercised
that path, which is why the whole suite passed throughout.

This was not a remote corner. The clause itself makes overriding a copy hook
grounds for refusal, and deep-copying such a subclass is precisely what leaves
the buffers writeable, so the two conditions coincide by construction rather
than by coincidence: deep-copying a subclass that overrides `__setstate__`
returns `0.7552125` against an exact `0.678150` on the `C = 2` fixture above,
the full `0.0770625`.

**Validity is a question about the object; eligibility is a question about the
route. The first is answered first.** Three consequences follow, none of them
optional:

- The coefficient read may no longer rely on the checks that now run after it.
  `__dict__` is one of the lookup hooks being compared, so `problem.__dict__`
  would be a lookup whose honesty has not yet been established — a subclass
  defining it as a property could hand the check frozen decoys and then be
  refused quietly a few lines later. Instead the read binds `AffineDynamics`'s
  own `__dict__` descriptor and calls it, the same narrowing `_defined_as`
  applies to class attributes and for the same reason.
- The three eligibility checks must still answer quietly for a *valid*
  problem. Each is paired with a control asserting that the same shape, with
  its coefficients intact, still returns `False`. Without those the reordering
  would be satisfied by a check that raises for every subclass.
- Running ahead of eligibility means the check can no longer borrow
  eligibility's conclusions, and it had been borrowing one. See below.

#### The invariant is conditional, so the check carries its own precondition

Coefficient immutability is not a property every `AffineDynamics` instance
owes. It exists because `AffineDynamics.F` and `G` hand back `self._M` and
`self._C` *by identity* and the tape retains them. (`f` returns a fresh array,
but reads all three buffers to build it, which is why an unfrozen `_b` still
matters even though nothing hands it out.) An instance holding none of that
storage has nothing the tape can alias.

Gating the check on `isinstance(problem, AffineDynamics)` alone assumed
otherwise and refused two working shapes outright:

| Shape | Was | Should be |
|---|---|---|
| Subclass overriding `f`, `F`, `G`, not calling `super().__init__` | raises | quiet `False`, exact gradient on the general route |
| `class C(TimeVaryingAffineDynamics, AffineDynamics)` | raises | quiet `False` per [C-16.2](../../NUMERICS.md#c-16) |

Both are correct general-route problems, and the second is a refusal this
contract already requires to be *quiet*. Raising there is a regression dressed
as a guard.

The precondition is therefore **class membership together with a flag the
root initialiser records about itself**: `AffineDynamics.__init__` writes
`_affine_root_initialised` into the instance dictionary after it has
established and frozen all three buffers.

Two inferences were written before it, and both are wrong *as replacements
for it*. The distinction matters, because one of them is right beside it.

**From the MRO** — "is any of `f`, `F`, `G` still the root's own definition?"
This is unsound in the quiet direction, because an override may delegate:

```python
class Instrumented(AffineDynamics):
    def F(self, y, u, t): return super().F(y, u, t)   # ...and f, G likewise
```

Three distinct function objects, every buffer read. The check was skipped and
the fixture below returned `0.7552125` against an exact `0.6781500`. **Method
identity cannot witness what a method reads.** Delegating wrappers —
instrumentation, unit conversion, logging — are ordinary, not adversarial.

**From presence** — "are `_M`, `_C`, `_b` there?" This evaluates descriptors
on classes that never had them. A valid general-route subclass may leave `_M`
as a property that must never run, and asking made the *check* raise. It also
misreads a subclass that reuses one of those names for its own storage as a
root instance taken apart after construction. A recorded flag asks nobody
anything. Presence returns below as a **widening**, where a descriptor that
refuses to answer hands out nothing instead of propagating: a name that
cannot be read aliases nothing, and a reader reaching for it during the solve
raises from the same descriptor, loudly.

#### The flag alone is not the whole precondition either

Taking "the initialiser did not run, so nothing is aliased" as *sufficient*
was itself a defect, and it was reproduced through the public optimizer. It
is true only of a subclass that also replaces the members which hand a buffer
back. One that assigns `self._M`, `self._C` and `self._b` itself and inherits
`f`, `F` and `G` runs the root's `return self._M` over writeable storage:

```python
class BuildsThemItself(AffineDynamics):
    def __init__(self, ...):
        self._M = ...; self._C = ...; self._b = ...    # no super().__init__
```

Nothing exotic — this is what a subclass looks like when its author assembles
the coefficients from a file or a mesh and sees no reason to route them
through `super().__init__`. No flag meant no check, while every reader was the
root's own; `affine_dynamics_verified` returned `True`, the affine route ran,
and the gradient was displaced by the full `0.0770625`.

So the precondition is the flag **or** the MRO question above, and the second
also answers *which* buffers are live rather than demanding all three: `F`
reads only `_M`, `G` only `_C`, `f` all three. A subclass overriding `f` and
`G` but not `F` owes `_M` and nothing else.
This is not the rejected inference reinstated. As a *replacement* it must
conclude "reads nothing" from an override it cannot see into, which is the
unsound direction. Used as a **widening** it only ever adds instances to the
checked set.

A second widening closes what method identity cannot see. A subclass that
both declines the root initialiser *and* delegates all three readers —

```python
class BuildsAndDelegates(AffineDynamics):
    def __init__(self): self._M, self._C, self._b = ...   # from a file, a mesh
    def F(self, y, u, t): return super().F(y, u, t)       # ...and f, G likewise
```

— has no marker and no reader of its own, so the live set was empty while
`super().F(...)` handed out its writeable `_M`: the full `0.0770625`
displacement through the public optimizer, with every guard quiet. Each half
of that shape is ordinary alone, which is the whole difficulty; the
combination is not exotic. So **any root coefficient name that resolves at
all is live** as well. That is the same question `__setstate__` asks of a
restored object, asked here for the same reason: the storage is what the tape
will alias, whatever the class says about who reads it.

It is **resolution, not type**, that makes a name live, and that was a cure
rather than a choice. Obliging only names holding an `ndarray` left the same
delegating shape holding three mutable `csr_array` coefficients — an ordinary
choice for a large linear system — outside both widenings, and it returned
the same `0.0770625` displacement through the public optimizer, quiet.
Whether the storage is usable is decided *afterwards*, by the refusal below,
which raises; deciding it inside the widening is skipping under another name.
The two directions are not symmetric: obliging a name that turns out fine
costs a refusal the caller can read and lift, while declining to oblige one
costs a wrong gradient nobody sees.

Both widenings are still load-bearing, and what identity adds is **a name
that does not resolve at all**. A subclass inheriting `F` while holding no
`_M` is refused by name at the route decision rather than raising
`AttributeError` from inside the first step.

And the flag remains load-bearing beside both, for the one thing neither can
do: **remember**. Each widening asks what the instance currently is; only the
flag knows that three frozen arrays were established here and that two are
left. A subclass that delegates its readers and then *deletes* `_M` is
invisible to identity (three overrides) and to presence (no name to resolve),
and is refused by name only because the marker survives. The three mechanisms
overlap almost everywhere, which is why each needed a test built so the other
two cannot reach it; see the note under the injection table.

Only `f`, `F` and `G` count for the identity half. `M`, `C` and `b` return by
identity too, but nothing on the solve path reads them.

The consequence for an instance holding *no* buffers is that it is now
refused by name at the route decision rather than raising `AttributeError`
several frames later. That is the same rule reaching a harmless instance, and
it is accepted rather than carved out by a second judgement about which
absences matter.

The flag is read through `AffineDynamics.__dict__["__dict__"]`, not by
attribute access, because unlike a coefficient it is not something a subclass
is entitled to answer for — one defining `__dict__` as a property returning
`{}` would otherwise hide it and silence the check. The buffers are read the
opposite way, and the next section says why.

**The MRO question is asked of `type`, not of the class.** `cls.__mro__` and
`klass.__dict__` are both attribute accesses on a class object, so a
*metaclass* answers them. One returning an MRO without `AffineDynamics` in it,
or a class dictionary reporting `f`, `F` and `G` as the subclass's own, made
the widening conclude that nothing was live; the flag was absent because the
root initialiser had not run, and the full `0.0770625` displacement went
through the public optimizer with every guard quiet. The readings are taken
through `type.__dict__["__mro__"]` and `type.__dict__["__dict__"]`, which
resolve on the type itself and cannot be intercepted. This is the previous
paragraph's generalisation applied one level up: a metaclass is to a class
what a `__dict__` property is to an instance.

**The flag is written through the same descriptor it is read through.** It
was not, and reading defensively while writing trustingly protects nothing: a
subclass exposing a different mapping under `__dict__` took delivery of the
marker there, the guarded read went to the real instance dictionary and found
nothing, and all three call sites concluded the root initialiser had never run
and fell silent — `0.7552125` against an exact `0.6781500` on the fixture
above. A guarded read only establishes what it claims if every write it
inspects went to the same place. This generalises beyond this clause: wherever
a check reads state through a path chosen to be unforgeable, every producer of
that state must use that same path.

**A live name that is not a frozen `ndarray` raises.** It does not matter
which mechanism made it live: once something answers for `_M`, whatever
answers is what the tape may alias, and storage whose aliasing cannot be
reasoned about — a `scipy.sparse` array, a `memoryview` — is refused by name.
Skipping was the earlier behaviour and it failed in the silent direction: an
unreadable buffer was treated as an absent one, the precondition concluded
there was no root storage, and the problem verified as affine with its tape
aliasing it unchecked. The message distinguishes the rooted case (three
frozen arrays *were* established here and one has been replaced since) from
the unrooted one (this is one of the root's coefficient names and something
on this instance answers for it), because the remedies differ.

The cost of not inferring liveness is over-refusal, and it is paid by exactly
one shape: a subclass that replaces all three readers and keeps writeable
storage of its own under one of the root's buffer names. It is refused,
loudly, although it works. That trade is deliberate — a loud refusal of a
working problem, against never staying quiet about a broken one — the refusal
names the attribute that collided, renaming it lifts the refusal, and
restoration already charged this shape the same price by freezing that name on
every copy. A rule that binds on copying but not on checking is not one rule.
It is pinned by a test so that narrowing it later is a decision rather than an
accident.

#### The buffer is read the way the method reads it

`AffineDynamics.F` is `return self._M`, so `getattr(problem, "_M")` resolves
what `F` resolves. Two more-defensive-looking reads were tried and are both
wrong:

| Read | Fails on |
|---|---|
| `problem.__dict__[name]` | a `__slots__` buffer (absent from the dict); a subclass `__dict__` property |
| `AffineDynamics.__dict__["__dict__"].__get__(problem)` | a `__slots__` buffer; a `_M` **property**, where the dict still holds the frozen array `__init__` wrote while `F` returns the writeable one the property computes |

The second is the instructive failure. It was chosen precisely to bypass the
subclass's attribute machinery — and bypassing it meant reading storage that
the machinery had displaced, so `affine_dynamics_verified` returned `True` on
a problem whose gradient was displaced by `0.0770625`. A more suspicious read
is not a more faithful one. Plain attribute access resolves slots, properties
and descriptors exactly as `F` does; a subclass `__getattribute__` that lied
would have to lie to `F` too, and then the lie *is* what gets retained.

This is faithful for any descriptor that answers **consistently**. One that
returns a frozen array to the check and a writeable one to the next read
defeats it, as would any number of other things Python permits at runtime; see
the [current envelope](../../NUMERICS.md#c-15-7-envelope).

**Restoration resolves storage by the same rule.** `__setstate__` read the
buffers out of the instance dictionary, so a subclass declaring
`__slots__ = ("_M",)` — which verifies, solves correctly, and is explicitly
within the envelope — raised `KeyError('_M')` on every `copy.copy`,
`copy.deepcopy` and pickle round trip. Certifying a shape for one obligation
of this clause while another rejects it is the inconsistency, not the
exception: the rule is one rule, applied wherever a coefficient is reached.

**Literally the same rule, not the same wording.** Two spellings of one
question drifted apart once already. The check absorbed any exception a
coefficient descriptor raised; restoration used `getattr` with a default,
which absorbs only `AttributeError`. A general-route subclass whose `_M`
property raised anything else was therefore accepted by the optimizer — it
returns the closed-form gradient — and could not be copied, deep-copied or
pickled at all. The converse asymmetry is the dangerous one: obliging only
names holding an `ndarray` during restoration let a delegating subclass with
`csr_array` coefficients, which the guard refuses, round-trip quietly into a
writeable copy. Both now resolve names through one function -- **every** read
of a coefficient, including the guard's own last one, which restated the rule
as a bare `getattr` and so let a refusing descriptor's exception out of the
route decision in place of the `MissingCoefficients` naming the buffer.
Restating a shared rule is how it stops being shared.

The one half restoration does *not* take from the check is reader identity. A
name an inherited reader obliges but which resolves to nothing cannot be
conjured during restoration, so it stays the route decision's to refuse.
**Restoration freezes what resolves; the guard obliges what must.**

**And the freeze is a second pass over re-read values.** `setattr` runs a
subclass property setter. One that stores a copy of what it was handed leaves
the caller's name bound to the object that was discarded, so the freeze lands
on that one and not on the array `F` will return; one that rewrites a
*different* coefficient replaces a buffer already frozen earlier in the same
loop. Both came back from an ordinary pickle round trip owning and *writeable*
on a subclass that verified before it. Nothing is frozen until no coefficient
setter can still run. Acting on a value fetched by a path other than the one
that will be read is the mistake this clause keeps making, in each of its
parts.

**The replacement is nevertheless frozen before the setter sees it.** That is
a second claim, not a weakening of the first. The second pass asks what the
object resolves to once every setter has run, so it only ever reaches the
owner; but a setter may derive a *second* handle from its argument — a view, a
reshape, a slice kept for fast access — and `writeable` is a per-array flag
that does not reach views already made from an array. A handle taken from a
writeable replacement therefore stayed writeable over the coefficient's own
memory while the owner was frozen behind it, and the guard, which re-reads
`_M`, was answered by the owner and saw nothing wrong. A default-protocol
pickle round trip of such a subclass verified and returned `0.7552125` against
an exact `0.6781500` on the fixture above. Only that route reached it: NumPy
reconstructs the array as a view over the pickle buffer, which does not own its
data, while `deepcopy` and protocol 4 hand back an owning array that a rooted
restoration leaves alone. Handing the setter a read-only array closes that at the source
*for a handle derived from that array*, a view of a read-only array being
read-only. It does not reach a setter that allocates storage of its own, which
is the next rule.

**And a coefficient must be the array the root froze.** This is the rule that
makes the freeze mean anything, and it is about provenance rather than state.
Everything above arranges for the array a coefficient resolves to be frozen. A
subclass setter is ordinary code and may store an array of its own; if it does,
it can derive a handle while that array is still writeable and then match the
argument's `writeable` flag onto the copy it keeps. The result satisfies every
check this clause can make — the coefficient resolves to an owning, frozen
array — while the subclass holds a writeable alias of the memory the tape
holds. Measured on the fixture above at `C = 2`, from plain construction, from
`copy.copy`, and from a default-protocol pickle: the problem verified and the
optimizer returned `0.7552125` against a closed-form `0.6781500`.

No ordering reaches it. The constructor and restoration both freeze before they
assign, but the setter chooses which memory becomes the coefficient and may
choose memory it has already aliased. So the coefficient is required to *be*
the array the root froze, checked by identity after assignment in both places —
`__setstate__` does not run `__init__`, and without its own check a pickle
reintroduces the subclass the constructor turns away.

**What that closes, and what it does not.** Once the coefficient is the array
the root froze, NumPy refuses to make a writeable view of it — by `view`,
`as_strided`, `reshape` or `frombuffer` alike — so a setter cannot derive a
writeable handle from what it was handed. It *can* unfreeze that array first,
take the handle, and freeze it again before storing it; the result is
indistinguishable and is not reached. That is the excluded class this clause
already names, occurring a few lines earlier than the usual example, and it is
listed in the [current ENVELOPE](../../NUMERICS.md#c-15-7-envelope) with its measurement. The rule closes
*substitution*, which was reachable without anyone intending it. It does not
close sabotage, and a raw `ctypes` pointer — which defeats a C++ `const` just
as completely — is outside the envelope for the same reason.

Nor does it reach a subclass that never calls `super().__init__`. There is no
root array to compare against, and requiring one would refuse every unrooted
general-route problem that happens to hold these names, a shape this clause
deliberately supports. Such a subclass owns its coefficients outright; its
buffers are still checked for being frozen and owning, and their provenance is
its own affair.

The check runs after all three coefficients are assigned, not after each, in
the constructor and in restoration alike. One coefficient's setter may rewrite
another, and a subclass that stores `_C` faithfully while rebuilding `_M` on
the way past passes a per-assignment check on every name and still ends up
holding a substituted `_M` — measured through both paths, and the restoration
half was written the wrong way round first.

**And the expectation is recorded for every name, not only the ones assigned.**
Restoration does not replace a coefficient that already satisfies the
constructor's postcondition, and recording an expectation only where it
replaced one left the skipped names unexamined. A subclass whose `_C` arrives
as a view — so it must be replaced — and whose `_M` arrives owning and frozen —
so it is left alone — can rebuild `_M` from inside `_C`'s setter and keep a
writeable view of the rebuild. `OBSERVED` on the fixture above at `C = 2`,
through ordinary construction and `copy.copy` with nothing patched and nothing
deliberately unfrozen: the copy passed every check restoration makes, and the
optimizer returned `0.7552125` against a closed-form and independently
referenced `0.6781500`. Being skipped is not being exempt; it only means the
expectation is the array that arrived rather than the one restoration made.

**And the names to check are not final until the last setter has run.** They
were enumerated once, before the assignment loop, so a name that was *absent*
at that moment was absent from every later check. An unrooted instance can
gain one: a setter that reconstructs a coefficient the arriving state did not
carry produces a buffer the root never froze, holding whatever handle the
setter kept. `OBSERVED`, same fixture and same displacement, with every check
quiet. Restoration re-enumerates afterwards and records an absent name as
having been *expected to stay absent*, which is what makes the refusal true
rather than merely convenient.

The refusal distinguishes four cases, because the remedy does, and *whether a
name was assigned* does not decide between them. A setter can store its own
argument faithfully and still be holding a substitute at the end, because a
later setter replaced it. So the refusal separates: its own setter substituted;
its own setter was faithful and a later one replaced it; it was never assigned
and another setter replaced it; and it did not exist when restoration began.
Naming any of these as another sends the caller to a setter that is behaving
perfectly, or to one that was never called.

**Nothing is left writeable while a setter can still run.** Restoration used to
skip a rooted instance's buffer whenever it owned its data, on the reasoning
that a rooted instance's arrays were frozen already. They are not: `deepcopy`
and pickle protocol 4 rebuild a coefficient as an owning but *writeable* array,
and the pass that freezes it runs after every setter. The buffer therefore
stayed writeable for exactly as long as the other coefficients' setters were
running, and one of them took a view of it — `OBSERVED`, same fixture, same
displacement, with the finished object indistinguishable from a correct one:
the owner frozen, its provenance intact, and the view reachable from no
coefficient. Only the ordering was wrong. The rule is now uniform — anything
not already at the postcondition is replaced by a frozen copy before any setter
runs — which costs one extra copy on the `deepcopy` and protocol-4 routes and
removes the last moment at which a live coefficient is writeable.

**And *every* copy is taken before the *first* setter runs**, not one name at a
time. Interleaving them satisfied the rule for the name being assigned and left
the names after it still holding the writeable arrays `deepcopy` and protocol 4
hand back. A setter writing into one of those — `self._C[...] = 3.0` from
inside `_M`'s setter — wrote into the array restoration had not copied yet, so
the change was carried into the replacement as though it had arrived that way.
`OBSERVED` through `copy.deepcopy` of the fixture above built at `C = 2`: the
copy held `C = 3`, owning and frozen, with its provenance intact and
`affine_dynamics_verified` returning `True`, and it then differentiated exactly
— for the coefficient nobody asked for, `[1.1265375, 1.0241250]` against the
source's `[0.6781500, 0.6165000]`. This is the ordering error in its second
shape: the first version checked at the wrong moment, and the second read at
the wrong moment.

**And being at the postcondition is not the same as having been put there by
the constructor.** A buffer arriving owning and frozen was taken as already
correct and left alone. But if the subclass produced the state, it decides
what arrives: `__getstate__` can allocate an array, take a writeable view of
it, freeze the array, and return both. What restoration is then shown
satisfies every property it checks, and nothing about the finished object
disagrees — the flags are right, the alias is reachable from no coefficient,
and identity confirms itself, because the fabricated array is the only one
restoration ever saw. `OBSERVED`, same fixture and same displacement, through
`copy.copy` with nothing patched and nothing unfrozen.

So a buffer at the postcondition is left where it is only when the root can
**vouch** for it. `AffineDynamics.__getstate__` records, in the state it
returns and under a key of its own, the array each coefficient name resolved to
at the moment the state was produced — or `None` for a name the state does not
carry, which is how a coefficient answered from storage outside the state is
distinguished from one missing altogether. Restoration lifts that record out
before anything else reads the state, and leaves a coefficient alone only when
the array that arrived **is** the array recorded, by identity.

Identity survives the honest routes because `deepcopy` and `pickle` memoise by
identity: an array referenced from both the state and the record is copied once
and referenced twice, so the correspondence is carried across rather than
asserted. A substitution in `state["_M"]` breaks it, and so does taking the
root's own state and replacing an entry in it.

**But identity says where an array came from, not what was done to it on the
way**, and the exemption needs both. A copy arrives *writeable* — the flag is
restored only afterwards — so anything else in the state graph copied during
that window may take a view of it and keep that view after the owner is frozen
again. Memoisation, which is what carries the record across, carries the record
across for the aliased copy too, so identity agrees and the exemption is
granted. `OBSERVED` on the fixture above at `C = 2` with **no subclass at
all**: a plain `AffineDynamics` carrying an ordinary attribute that keeps a
flattened view of `M` and rebuilds it in its own `__deepcopy__` verified before
and after `copy.deepcopy`, the copy's `_M` owning and frozen, a writeable alias
sharing that memory, and `0.7552125` against an exact `0.6781500`.

So the exemption is granted only to a state that was **handed on unchanged**.
The record carries a marker under a key no attribute can collide with:
`copy.copy` passes the mapping's values through, so it arrives as itself, while
`deepcopy` and every pickle protocol reconstruct it. Nothing that was not
copied can have been aliased while it was being copied; a rebuilt state has its
coefficients replaced, which severs any view the traversal left. That costs
nothing on a rebuilt state — the state is the traversal's own, so no setter it
runs can reach the object being copied — and on the ordinary deep routes it
changes nothing at all, because `deepcopy` and protocol 4 hand back writeable
arrays and protocol 5 a non-owning one, none of which was ever exempt. It binds
exactly where something re-froze the array mid-traversal, which is the case
that needs it.

A coefficient the state does not carry keeps its exemption on either route: it
was not copied, so no traversal can have aliased it. That is the `None`
sentinel's second job, and a read-only property answering for a coefficient
depends on it, since assigning one raises.

**What "does not carry" means is reachability, not membership.** Written as
"the array is not a value of the instance dictionary" it is a much weaker
statement, and two supported shapes fall through the gap: a coefficient
answered from inside an ordinary attribute object, and one held in a
`__slots__` mapping, which is reported separately by `object.__getstate__` and
appears in the instance dictionary not at all. Both were recorded as never
having travelled, kept the exemption on a rebuilt state, and came back with a
companion holding a writeable view of the frozen coefficient — `OBSERVED` on
the fixture above at `C = 2` through `copy.deepcopy` of each, and through
pickle protocols 0 to 4 for the first. So the record asks what a traversal of
the whole produced state — both halves, through containers and through objects
— could reach, and a view's owner counts as reachable through the view, since
freezing a view leaves the owner writeable.

Where that walk cannot see inside something, it reports so and the coefficient
is treated as reachable: a replacement that may not have been needed, and a
loud refusal where the replacement cannot be stored.

**An object that redefines its own copying makes the walk inconclusive**, and
that branch has a witness. The walk predicts a traversal from the graph as it
stands when the state is produced; an object carrying `__deepcopy__`,
`__reduce__` or any other traversal hook is not predictable from that graph,
because the hook may build something the graph does not contain. A companion
that is *empty* when the state is produced, and whose hook then allocates the
coefficient from storage outside the state, takes a flattened view while the
fresh array is still writeable and freezes the array behind it, defeats a walk
that is correct about the state and silent about the copy. Nothing on the
problem class is overridden, so the hook checks in verification have nothing
to object to. `OBSERVED` on the fixture above at `C = 2`: the copy verified,
its `_M` owning and frozen, and the companion holding a writeable alias of it.

So this branch is a cure with a measurement behind it, not the conservative
default it was first recorded as. Treating the coefficient as reachable turns
that case from a quiet wrong answer into a named refusal, which is the correct
outcome for a shape whose coefficient is manufactured by its own duplication.

The walk's *scope* then has to match the traversal's, and eight corrections
were needed before it did. Each was measured the same way — the hiding place
restored one at a time, with the others cured — and each on its own gave a
verified copy holding a writeable alias of its own frozen `_M`, and the
displaced `0.7552125` against an exact `0.6781500`:

- A **container subclass** was matched as its builtin and its contents walked,
  but not its own attributes, so a `dict` subclass holding the companion in an
  instance attribute was invisible.
- **Slots were replayed from the `__slots__` declaration**, which for the legal
  bare-string form `__slots__ = "payload"` iterates seven single characters and
  finds no slot at all. They are now read from the member descriptors the class
  actually defines, which is what the names resolve to, and which also handles
  mangled private names.
- A class-level **`__getattribute__`** can answer a request for `__deepcopy__`
  with a hook that appears in no class dictionary. A copying route asks by
  lookup, so the presence of a lookup hook is itself disqualifying. This is not
  the excluded instance-bound reducer: the behaviour is on the class, and the
  coefficient descriptor still answers the same way every time.
- **Atomicity was decided by `isinstance`.** `int` is copied by returning it,
  but a subclass of `int` carries an instance dictionary that an ordinary copy
  reconstructs. Atomicity is a property of the exact type.
- A **subclass of any modelled type that declares anything of its own** need
  define no listed hook at all. NumPy calls an `ndarray` subclass's
  `__array_finalize__` while building the new array; a `dict` subclass is not
  handed its storage whole but rebuilt blank and repopulated item by item, so
  its `__setitem__` runs on the copy. Either is enough to allocate a
  coefficient and keep a writeable view of it, and neither name appears in
  any list of copy-protocol hooks. So the question asked is not *which* names
  a subclass defines — asking by name is what failed repeatedly above — but
  whether it defines any, since a subclass declaring nothing is rebuilt
  exactly as its base is and can materialise nothing the walk cannot already
  see. Both halves are load-bearing: restricting the rule to `ndarray` leaves
  the `__setitem__` witness undetected, and removing it leaves both.

  Exactness alone was tried first and was too wide in the costly direction. A
  problem answering `_M` from a read-only module-level buffer cannot have that
  coefficient replaced at all, so withholding its exemption is a refusal
  rather than a copy — and an inert `list`, `dict` or `ndarray` subclass held
  as incidental metadata was enough to trigger it. `OBSERVED` on the fixture
  above: each of the three lost every rebuilding route to
  `MutableCoefficients` naming `_M`, while the same problem without the
  metadata answered exactly.
- **`__new__` runs on every rebuilding route** — `copy` and `pickle` alike make
  the object by calling it before they put any state in — so a companion
  defining one of its own materialises with nothing else overridden. What it
  builds has to be held off the instance to witness this, since the state
  applied afterwards would overwrite it; that detail cost one iteration of the
  witness, which passed for the wrong reason until it was found.
- An **object-dtype array** is an exact `ndarray`, so no subclass rule reaches
  it, and it is a container of Python objects that NumPy copies one by one.
  Nothing about its shape or dtype says what they are, so it is reported
  inconclusive rather than walked element by element: the elements of a
  structured dtype are reached only through its fields, and guessing a
  traversal is the error the entries above record. A coefficient is refused an
  object dtype outright, so only a companion reaches this.
- A **mapping's own `keys` and `values`** describe whatever view of itself it
  likes; the storage a reducer copies is the one underneath. Contents are read
  through `dict.keys` and `dict.values` for that reason. This one is now
  subsumed: any inexact `dict` subclass is inconclusive before its contents
  are read, and for an exact `dict` the two spellings agree, so the rule is
  retained as defence in depth with no injection able to witness it.

A class namespace turned out not to be a fixed property of the class, which
is this clause's recurring error in its sharpest form. `copyreg` caches
`__slotnames__` **into the class** the first time an instance of it is copied
or pickled, so a rule reading a namespace answers differently before and
after the first duplication. `OBSERVED` on the fixture above: the first
round trip of a problem carrying inert metadata answered exactly and every
later one raised `MutableCoefficients`, the metadata's class having acquired
a name in between. Names written by something other than the class's author
are therefore excluded by name, and the assertion held is that the answer
does not depend on the round trip's ordinal — which nothing weaker than
repeating the round trip can check.

<a id="storage-is-not-reconstruction"></a>

**Storage is not reconstruction.** The walk does not follow what an array
shares memory with. It once followed `ndarray.base` and, through it,
`memoryview.obj`, on the stated theory that "a view travels with whatever
owns its memory". That theory is false, and is now pinned by a test:
`OBSERVED` for a dictionary carrying an array and a view of it, `deepcopy`
and in-band pickle protocols 0 through 5 each returned a pair that shares no
memory, because each array is reconstructed independently and no `.base`
relationship is serialised; `copy.copy` returns the very same two objects
and rebuilds nothing. An owner reached only through a view is therefore
never rebuilt, and nothing it declares about its own copying is ever called.

The scope is the **default** routes. A faithful out-of-band load borrows the
buffers it is handed, and so can preserve sharing between the restored pair
and with the source — `OBSERVED` for the same dictionary dumped with
`buffer_callback` and loaded from those buffers. It still does not rebuild
the exporter or call its hooks, which is the property this walk depends on;
what it shares is the source's own storage, and a verified source's
coefficients are frozen. A *replacement* provider shares the caller's
storage instead, and the [current ENVELOPE](../../NUMERICS.md#c-15-7-envelope) records what that costs a
coefficient the walk sees only through an exporter.

The claim is about *sharing*, not ownership. In band, protocol 5 hands a
contiguous array back as a **non-owning** view of an incidental array it
made while restoring — `OBSERVED` for C-contiguous, Fortran-ordered and
`np.frombuffer` arrays alike, each restored with `owndata` false and a base
chain ending in a `memoryview`, and each sharing no memory with its source.
A strided slice was restored owning its storage. So "the copy owns its
bytes" is false and "the copy shares with nothing that travelled beside it"
is what holds.

Following the chain anyway cost answers in both directions available to it.
It recorded owners as *carried* that no copy would rebuild, which forces a
replacement the coefficient may have no setter for; and it judged whatever
it found at the end of the chain by hooks no copy would consult. `OBSERVED`
on the fixture above with metadata built by `np.frombuffer(array("d", ...))`
— an ordinary way to read a numeric array, whose base chain ends in a
`memoryview` over an `array.array`, and `array.array` defines
`__reduce_ex__`, `__copy__` and `__deepcopy__`: `deepcopy` and pickle
protocols 4 and 5 all raised `MutableCoefficients` naming `_M`, the
module-level frozen array a read-only property answers with and nothing can
replace, for a problem whose coefficient the metadata has nothing to do
with.

The earlier symptom was the same mistake seen from the other side. A
problem answering from an unassignable buffer survived its first protocol-5
round trip and was refused on its second, because protocol 5 restores a
contiguous numeric array as a view whose base chain ends in a `memoryview`
that the walk could read nothing from and so called opaque. Modelling
`memoryview` cured that symptom and is now withdrawn with the chain itself:
with storage links unfollowed, a `memoryview` can reach the walk only by
being carried in the state directly, and `complete` decides nothing on that
route — a shallow copy passes the state through, so a coefficient already at
the postcondition is vouched for and left alone, while no *default* deep
route can copy a `memoryview` at all, Python raising before this library is
consulted. A registered reducer can, and is read as inconclusive on its own
grounds below, whether or not the type is modelled.
A guard whose removal the injection campaign cannot detect is not kept.

**A namespace is not where all of the answer is.** A reducer registered
through `copyreg.pickle` sits in a module-level table that `deepcopy` and
`pickle` consult by exact type and appears in no class namespace at all, so
a scan of namespaces cannot see it — and registering one is an ordinary use
of the copy protocol rather than tampering. It reaches the *modelled* types
in particular, whose own hooks the scan skips on the grounds that their
copying is understood. `OBSERVED` on the C-15.7 fixture at `C = 2` with a
reducer registered for `memoryview` that rebuilds a view and, while doing
so, allocates the coefficient and keeps a writeable handle on it: the deep
copy verified with `_M` frozen and owning, the handle shared its memory and
was writeable, and a write through it in the terminal derivative callback
moved the first gradient component from `0.6781500` to `0.7552125`. An exact
type present in `copyreg.dispatch_table` is therefore inconclusive before
its namespace is read at all. The table is keyed by exact type, so a
subclass of a registered type is neither covered nor condemned by the entry.

The registration is asked **once**, of every object the walk reaches. A
second copy of the question inside the namespace scan, added when this was
first cured, became unreachable in effect once the walk asked it ahead of
atomicity, and the C-15.7 campaign reported it as a defect no test could
detect. One site states the rule; the namespace scan states the other half,
further down, where only the objects that have a namespace arrive.

The cost is the one a written hook already carries. NumPy registers a
reducer for `ufunc`, so a problem answering from a buffer nothing can
replace and carrying `np.sin` beside it is refused — exactly as one carrying
a `datetime.date` is, whose `__reduce__` is written on the type itself and
which this clause has always refused. Where a hook is kept does not change
what it can do, and neither refusal is silent.

**Atomicity is a statement about the traversal, not about the type**, so the
registration is read *before* it. The standard library has registered
`pickle_complex` for `complex` since long before this library existed, and
`copy.deepcopy` answers a complex number from its own dispatch table without
ever consulting `copyreg` — but every pickle protocol consults it. A walk
that skipped an atomic value before asking what was registered for it
therefore read the deep copy correctly and the round trip not at all.
`OBSERVED` on the C-15.7 fixture at `C = 2` with the standard reducer
replaced by one substituting the value while allocating the coefficient
behind it: the protocol-5 round trip verified with `_M` frozen and owning,
the handle shared its memory and was writeable, and the first gradient
component moved from `0.6781500` to `0.7552125`.

What keeps an ordinary complex number answerable is therefore not its
atomicity but that this walk **models that one registration**, comparing the
registered reducer with `copyreg.pickle_complex` by identity.
`pickle_complex` returns `(complex, (real, imag))` and can rebuild nothing
else, which is a fact about that function and not about the type: replacing
it costs the answer, loudly.

Distinguishing a constructor slot from a written `__new__` was tried first
and is **refuted**. Every type implemented in C exposes `__new__` as a
`builtin_function_or_method` while a `__new__` written in a class body is a
`staticmethod`, which is a statement about spelling and not about what the
object does. `__new__` may be *assigned* after the class exists, and an
assignment keeps whatever type the object already had: binding a built-in
method there leaves a namespace indistinguishable from a constructor slot
and an object that hands back one prepared in advance. `OBSERVED` on the
C-15.7 fixture at `C = 2` with that discrimination in place — a companion
whose assigned `__new__` returns an instance holding a writeable view of the
coefficient, reached by `copy.deepcopy` — the copy passed
`require_immutable_coefficients` with `_M` frozen and owning, and a write
through the retained view in the terminal derivative callback moved the
first gradient component from `0.6781500` to `0.7552125`. So `__new__` is
disqualifying whenever it appears, and the exemption for an ordinary
restored array is bought by modelling the one type that needed it rather
than by a rule about how a hook is spelled.

**A route does not hand back what it was given.** Every duplication test in
this clause built each case from a fresh source, so no test copied anything
a route had produced, and the finding above was invisible for that reason:
protocol 5's own output is the only input that carries a `memoryview`. `OBSERVED` on the fixture above with a coefficient answered
from an unassignable module-level buffer: the first protocol-5 round trip
gave the closed-form first gradient component `0.3026375`, and the second —
and a deep copy of the first — raised `MutableCoefficients` naming `_M`.
Protocols 0 to 4 survived three chained links throughout. Duplication
evidence therefore chains each route three deep from its own output, and
mixes kinds, since the kinds hand back different things.

The first two are what made the earlier disposition wrong. It had been argued
that neither was separately observable, on the grounds that the aliasing holder
must hold the array or a view of it and so is reachable anyway. That is true of
the *array* and false of the *companion*: the companion can be empty, and it is
the companion's presence that decides whether the walk is conclusive. Putting
the companion behind the omitted path — and, for the slot case, giving its
holder an inherited empty `__dict__` so the walk believes it has something to
look inside — makes each decide the outcome alone. The lesson is the session's
recurring one in yet another form: reasoning about what the walk would find,
rather than running it.

The inconclusive branch must cost a *copy* and never an *answer*, and that
direction is asserted rather than assumed. The list of things the walk cannot
predict grew once per finding above, and an over-broad entry is silent: it
refuses nothing and breaks no test, it merely stops the exemption ever being
granted. One such widening was made here — every class with instance
attributes carries a `__dict__` getset descriptor, so treating `__dict__` as a
lookup hook disqualified almost everything — and it went unnoticed until it
masked the `__array_finalize__` cure, whose reverted-cure run still refused. A
supported problem carrying ordinary incidental state of each awkward kind is
therefore required to verify, own a frozen coefficient and give the closed-form
gradient through every duplication route.

A class namespace is read through `type.__dict__["__dict__"]` rather than as
`klass.__dict__`, so a metaclass cannot answer the question with something
other than the class's own namespace, and the MRO likewise. A `__dict__` that
is not the ordinary getset descriptor is disqualifying for the same reason as a
lookup hook, since it decides what the walk gets to read; no witness is known
for that one, and it is recorded here as a conservative default rather than a
measured cure.

Two cheaper rules were tried first, and both asked about the wrong thing. Each
attempted to infer from the class whether the subclass had produced the state,
by asking which copy-protocol hooks it overrides — `__setstate__`,
`__getstate__`, `__deepcopy__`, `__copy__`, `__replace__`, `__reduce__`,
`__reduce_ex__`.

The list was too narrow, twice. First it omitted `__copy__` and `__deepcopy__`,
on the reasoning that they bypass `__setstate__` altogether; then it omitted
`__replace__`. They bypass only the reconstruction the protocol would otherwise
perform; any of them may reconstruct with `__new__` and call the inherited
`__setstate__` with a state it chose. So may an overridden `__setstate__`
delegating to `super()`, and so may a class overriding none of them: `copy`
obtains `__reduce_ex__` with `getattr`, which a class-level `__getattribute__`
answers. `OBSERVED`, same fixture and same displacement, through five such
routes.

And the list was also too **wide**, which is the direction that has no repair.
A class defining only `__replace__` — a hook `copy.copy` never calls — was
judged to have produced a state it had no part in, so its coefficients were
replaced and its setter ran. Under `copy.copy` the state objects are *shared
with the source*: the setter followed a reference `copy.copy` had carried
across and rewrote the source's own `_C`. `OBSERVED` on the fixture above,
nothing patched and nothing fabricated: copying the problem moved the gradient
of the object that was copied from `[0.6781500, 0.6165000]` to
`[1.1265375, 1.0241250]`.

No list can be right in both directions, because which hooks a route consults
is a property of the *route*, not of the class, and the class is all a static
list can see. A record written by the root at the moment the state is produced
is not a guess about either.

#### A coefficient is vouched for, not inspected

Everything above concerns what restoration does when it runs. A duplication
hook that never reaches restoration is not covered by any of it, and being
refused the *affine* route is not the same as being safe: `F` and `G` return
`_M` and `_C` by identity, the forward solve retains what they return, and the
general route reads them the same way. Ineligibility changes which solver runs,
not which array it reads.

So a hook of one's own, doing the plausible thing — deep-copy the state, keep a
handle on the matrix, freeze the buffers, put them on a fresh object — hands
back a clone whose `_M` is owning and frozen and whose retained handle is
writeable, because the clone's coefficients arrive writeable and the flag is
not something a copy carries. `OBSERVED` on the fixture above at `C = 2`: the
clone was refused the affine route, passed every check the arrays themselves
can answer, and gave `0.7552125` against an exact `0.6781500`.

Nothing about the clone's final state distinguishes it from an honest one. The
arrays are indistinguishable by construction, so validity stops asking them.
The root records, at the end of its own initialisation and again at the end of
its own restoration, the exact arrays it established; validity requires that
the coefficients presented are those arrays, by identity. A clone assembled by
a route that skipped both cannot produce that evidence and is refused aloud.

Three properties of the register are load-bearing:

- It is **module-level**, so no state carries it, no copy brings it along, and
  nothing written into a state can forge an entry.
- It is keyed by the **array**, not by the problem reading it, for the reason
  given below.
- It is checked **last**, after the flag checks. Those name a specific defect
  in a specific buffer; this one reports the absence of evidence, which is a
  weaker diagnosis and must not mask a precise one. Ordered first, it did.
- It applies **only to rooted instances**. A subclass that legitimately reads
  none of the root's buffers never enters the mechanism, and refusing it for
  not having used something it never touches costs an answer rather than a
  copy — the direction this clause is most careful about.

The evidence is held against the **array**, not against the problem that was
holding it. A frozen owning array cannot acquire a writeable alias afterwards
— a view of it is read-only, and cannot be made otherwise — so reading it from
a second object is exactly as sound as reading it from the first. Recording
the pair instead refuses a duplication hook that hands the clone the root's
own buffers, which manufactures nothing; `OBSERVED` on the fixture above,
`copy.copy` and `copy.deepcopy` of such a class raised `MutableCoefficients`
although `clone._M is source._M`, so a supported gradient could not be
computed at all. Pair evidence would also be lost the moment the original were
collected while the array it established was still in use. The distinction
that matters survives: a hook that *rebuilds* the arrays presents ones the
root never established and is still refused.

Entries are keyed by address, and each carries a weak reference used to
confirm identity before the record is trusted. That guard is against address
reuse and has no injection witness, being a race no test can schedule; it is
recorded here as a defensive check rather than a measured cure.

The alternative was to refuse any duplication override outright, which is
sound but coarse, or to record the shape as outside the envelope. Positive
evidence was preferred because it refuses exactly the objects that cannot
account for themselves, and it was adopted with the coarse refusal as the
declared fallback.

There is still nothing to establish for a vouched coefficient: those arrays are
the ones the constructor froze and already checked the provenance of, and
replacing one means assigning it. The rule was briefly written the other way,
replacing at the postcondition unconditionally, and that is what the
`copy.copy` measurement above was taken against. Note
what it does *not* turn on: whether the attribute happens to be settable.
Deciding by trying the assignment and excusing an `AttributeError` reopens the
fabrication hole in another shape — a read-only property over a fabricated
array — so a subclass with a read-only coefficient property *and* a state hook
is now refused aloud instead.

Leaving a coefficient alone is why an expectation is recorded for every name
rather than only the replaced ones: a coefficient left where it is, is one
another coefficient's setter can still substitute without restoration having
assigned it.

The cost is narrow and loud. A setter may **relocate** what it is handed — into
another key, a slot, under any name — and several supported subclasses do. It
may not store a **substitute**. Two subclasses this clause previously called
verified did exactly that, and they are kept as refusal cases: their difference
from the aliasing one is a single line, a view taken before the flag is
matched, which nothing in the object's final state can distinguish. A subclass
that wants to normalise its coefficients does so before `super().__init__`,
where the array is still its own.

**Which buffers are frozen is asked of the restored object, never of the state
it arrived in.** Three rules that read the state were tried, and a subclass
defeated each by shaping it:

| Rule | Defeated by |
|---|---|
| The restored marker | `__getstate__` returning only the coefficients |
| All three names present | the same override omitting `_b` as well, with `f` overridden so it is never read |
| Any coefficient name present | a property setter that writes the `_M` slot, the buffer arriving under another name |

Each reached `0.7552125` against an exact `0.6781500` on the fixture above,
through the public optimizer, with all three guards quiet — in the last case
on a buffer that was live and aliased by the inherited `F` while appearing in
no restored key. And no rule counting names could have survived anyway: a
one-name state is *indistinguishable* from a subclass legitimately reusing
`_M` for storage of its own.

The state is written by the subclass; the object is what the solve will read.
So **every coefficient name that resolves to an array on the finished instance
is frozen** — the same question the check asks, asked the same way, of nobody.
Where the marker survived, the root established all three and all three are
required, so anything else raises per [C-7](../../NUMERICS.md#c-7), and restoration records the
marker itself rather than passing on the one it was handed.

**Asking the object is not a passive read, and that is what makes it
sufficient.** A subclass can ship a coefficient under another name and rebuild
it lazily on first access, so nothing in the restored state is called `_M`. A
rule reading the state finds nothing to freeze and leaves the inherited `F`
handing out a writeable array. `getattr` instead runs whatever the object does
to answer — `__getattr__` included — so the buffer is built *there*, copied and
frozen, and the lazily-restored instance is indistinguishable from an eagerly
built one. The cost is that construction happens during restoration rather
than on first use; that is the trade this clause makes everywhere, because
what the solve will read is the only thing worth checking.

A name that never resolves at all is a different matter, and is not deferred
either: it is refused at the route decision **by name**, rather than raising
`AttributeError` from inside the first step.

Without the marker a buffer that is still writeable is **copied before it is
frozen**, even when it already owns its storage. `copy.copy` shares the array
with the original, so freezing in place reached back through that sharing and
froze the source, changing an object the caller never asked to change. The
marked path keeps sharing, because there both arrays are frozen already.

A buffer that is already owning **and** already frozen is left exactly where
it is, for the same reason stated the other way round: the freeze is a no-op,
so there is nothing to reach back to, and replacing it costs the only thing
this branch can lose. A read-only property answering for a coefficient cannot
be assigned to, and an ordinary round trip of such a subclass — accepted by
the optimizer, returning the closed form — raised `property '_M' ... has no
setter`. When the buffer is writeable *and* the storage refuses assignment
there is no safe outcome, since the copy cannot be stored and freezing the
original would change the caller's array; restoration then refuses, names the
buffer and says why, rather than letting a bare `AttributeError` out of
`copy.deepcopy`.

The copy is made with `np.array(buffer, copy=True, subok=True)`, not by
calling the buffer's own `copy`. `isinstance(..., ndarray)` admits subclasses
and the method is theirs to define: one returning `self` put the freeze
straight back onto the shared source this branch exists to protect, and one
perturbing the data moved a coefficient by `0.5` in silence — the
[C-7](../../NUMERICS.md#c-7) direction that must not be reachable. `np.array` dispatches to no
user-defined `copy`. It does run `__array_finalize__`, which no
subclass-preserving construction can avoid, and which is why restoration is
not where a subclass buffer is made safe — it is refused at the check.

`subok=True` rather than `subok=False`, because restoration must not silently
change what a buffer *is*. Normalising a `MaskedArray` to a base array
discards the mask, and a buffer the check is required to refuse would arrive
at it as one that passes. Restoration preserves faithfully so that the guard
can refuse loudly; converting here would be the guard certifying an object
that no longer exists.

The cost is narrow and loud: a subclass reusing a coefficient name for
writeable storage of its own finds that its copies cannot be written to, and
hears `assignment destination is read-only` if it tries. That is this clause's
standing trade — refuse a working problem audibly rather than stay quiet about
a broken one — and a test pins it on the fixture that pays it.

#### What the freeze can cover, and what it refuses instead

`writeable = False` seals the `ndarray` it is set on. It does not seal
anything that array merely carries or points at, and two refusals exist
because it does not.

An `object`-dtype array holds references, and freezing it stops the references
being rebound while leaving every referent mutable: the elements of a frozen
`dtype=object` `_M` can be assigned through after the check, and the tape's
aliased matrix changes underneath the adjoint. Reproduced through the public
optimizer at the full `0.0770625` displacement with the flag recorded and both
existing refusals satisfied.

An `ndarray` **subclass** holds state of its own that the flag likewise never
reaches. A `MaskedArray`'s mask is an ordinary writeable array; flipping one
entry mid-solve changed what the coefficient means, on a deep copy that
verified as affine with its data frozen and owning, and the gradient came back
`0.7706250` against an exact `0.6781500` on the fixture above at `C = 2`.
There is no general way to enumerate what an arbitrary subclass keeps.

Deep-freezing arbitrary referents is not available — there is no such
operation, and the referents may be any Python object at all. So both are
refused on inspection rather than certified: `dtype.hasobject`, and
`type(buffer) is not ndarray`, each raising `MutableCoefficients` naming the
buffer. The certified envelope is numeric coefficients, so this refuses
nothing [C-6](../../NUMERICS.md#c-6) admits, and it refuses *loudly* rather than certifying an
immutability the freeze cannot deliver. A guard must either establish its
invariant or say that it cannot.

The root initialiser refuses an `ndarray` subclass **argument** for the same
reason rather than converting it. `np.array(M, dtype=float)` would discard a
mask, so the caller's coefficient becomes a different matrix without anyone
being told. A list, tuple or scalar loses nothing by being converted and still
is. One rule: coefficients are base numeric arrays everywhere, and a buffer
that is not one is named, never normalised.

#### Where the check runs

`affine_dynamics_verified` is reached from `GLMOptimizer` only when the
structure has to be deduced, so a caller supplying an explicit
`ProblemStructure` bypassed the coefficient check altogether and received the
displaced gradient in silence — the same `0.7552125`.

Guarding `GLMOptimizer` is still not sufficient, because it is not the only
way in. `forward_solve`, `adjoint_solve` and `assemble_gradient` are exported
from `adjungo.stepping` and `adjungo.optimization` and compose into a gradient
without touching `GLMOptimizer` at all; that composition returned the same
`0.7552125`. The invariant belongs at the point of *retention*, and that point
is `forward_solve`: every step stores what `F` and `G` returned in the step
cache, and from there the tape aliases the buffers. Checking there covers
every composition that produces a trajectory, rather than every caller that
consumes one.

`require_immutable_coefficients` is called in three places for three different
reasons, and is public for that reason:

| Call site | Why |
|---|---|
| `affine_dynamics_verified` | validity before eligibility, so no quiet refusal pre-empts it |
| `GLMOptimizer.__init__` | refuses before any work, on both structure paths |
| `forward_solve` | the retention boundary; covers every low-level composition |

Three overlapping guards mean the outer two are individually invisible to any
behavioural test, and the first injection campaign proved it: removing the
`GLMOptimizer.__init__` call failed **zero** tests, because the regression
asserted `pytest.raises` around `GLMOptimizer(...).gradient(U)` as one
expression and `forward_solve` raised instead. The distinct claim — that the
refusal precedes any work — is only tested by constructing and *not* solving,
which is what that test now does. This is [C-15.5](../../NUMERICS.md#c-15) again: when a guard
is shadowed by a downstream one, assert it directly or it is decoration.

<a id="historical-campaign"></a>
## Injection evidence — `OBSERVED`

Under [R-11](../../NUMERICS.md#r-11) against an 822-test baseline with 0 failures, 95 of 95
injected defects detected. Counts are as emitted by the campaign:

| Injected defect | Failing tests |
|---|---|
| `__setstate__` removed entirely | 78 |
| `__setstate__` restores state but does not re-freeze | 77 |
| `__setstate__` leaves the buffers writeable outright | 113 |
| only `_M` is re-frozen; `_C` and `_b` are missed | 68 |
| `__setstate__` assumes a plain `dict`, breaking `__slots__` subclasses | 8 |
| `__setstate__` freezes without restoring ownership | 52 |
| verification stops checking the copy-protocol hooks | 8 |
| `__setstate__` dropped from the guarded copy hooks | 2 |
| `__deepcopy__` dropped from the guarded copy hooks | 1 |
| the mutable-coefficient refusal is removed | 30 |
| the refusal returns False instead of raising | 24 |
| the refusal checks writeable but not owndata | 1 |
| b is reshaped into a view rather than an owning copy | 64 |
| the freeze is dropped from the constructor as well | 143 |
| the refusal is ordered last again, behind the quiet checks | 17 |
| unreadable storage is skipped rather than refused | 10 |
| GLMOptimizer stops enforcing validity at construction | 2 |
| forward_solve stops enforcing validity at the retention point | 2 |
| the precondition is method identity alone, without the flag | 3 |
| the precondition reverts to probing for buffer presence | 12 |
| the root-initialised flag is read by attribute access | 2 |
| both widenings are dropped, leaving the flag alone | 17 |
| any inherited reader obliges all three buffers | 3 |
| F is dropped from the readers that oblige a buffer | 2 |
| `_defined_as` reads the MRO through the metaclass | 1 |
| the presence widening is dropped, leaving reader identity alone | 7 |
| presence obliges only names holding an `ndarray` | 4 |
| presence replaces reader identity rather than widening it | 4 |
| the `ndarray`-subclass refusal is removed | 2 |
| the root initialiser converts an `ndarray` subclass silently | 1 |
| the object-dtype refusal is removed | 1 |
| the defensive copy normalises away the buffer's subclass | 1 |
| the defensive copy dispatches to the buffer's own copy() | 2 |
| MissingCoefficients always claims the initialiser ran | 3 |
| the class gate is dropped, catching unrelated problems | 305 |
| buffers are read from the instance `dict`, not by attribute | 83 |
| the root-initialised flag is never recorded | 4 |
| the marker is written where it is not read | 1 |
| restoration reads coefficients from the instance `dict` | 65 |
| restoration freezes names the instance does not hold | 58 |
| restoration trusts the restored marker alone | 21 |
| restoration reads the restored keys, not the object | 11 |
| restoration does not record what it concluded | 1 |
| the guard restates the resolution rule as a bare getattr | 1 |
| a failed replacement always reports the buffer as writeable | 1 |
| restoration absorbs only AttributeError from a descriptor | 4 |
| restoration replaces a postcondition buffer whatever the record says | 52 |
| restoration takes a buffer at the postcondition on trust | 27 |
| restoration lets a failed setattr out as it comes | 3 |
| each coefficient is copied only when its turn comes | 4 |
| the record is trusted by name rather than by array identity | 3 |
| restoration does not ask about a delegating `__setstate__` | 2 |
| the record is written into the live instance dictionary | 5 |
| a coefficient the state does not carry is left out of the record | 47 |
| what travelled is decided by dictionary membership | 17 |
| the slot mapping is left out of the reachability walk | 1 |
| a rebuilt state is vouched for by identity alone | 10 |
| the untravelled marker is never recorded | 5 |
| the provenance record is never attached to the produced state | 52 |
| the replacement is frozen in one pass, before setters finish | 3 |
| the replacement is handed to the setter writeable | 4 |
| the constructor stops checking what the setter stored | 5 |
| restoration stops checking what the setter stored | 17 |
| restoration records an expectation only where it replaced | 5 |
| a replaced coefficient is reported as one that was not assigned | 1 |
| restoration blames a name's own setter for a later one's doing | 1 |
| the constructor blames a name's own setter regardless | 1 |
| restoration skips a rooted buffer that still owns its data | 29 |
| restoration does not re-enumerate the names afterwards | 1 |
| restoration checks each coefficient before the next is set | 3 |
| the provenance rule compares by value, not identity | 21 |
| the constructor checks each coefficient before the next is set | 1 |
| an unmarked buffer is frozen in place, reaching into the source | 33 |
| a copy hook does not make the walk inconclusive | 10 |
| a container subclass's own attributes are not walked | 2 |
| `slots` are replayed from the declaration, not the descriptors | 1 |
| a hook supplied by attribute lookup is not noticed | 1 |
| atomicity is decided by isinstance, so subclasses inherit it | 1 |
| a subclass of a modelled type is assumed to copy like it | 2 |
| an object array is walked as though it held no objects | 1 |
| an inconclusive walk refuses rather than replacing | 30 |
| only an `ndarray` subclass copies by rules of its own | 1 |
| `__new__` is not treated as a reconstruction hook | 2 |
| a subclass `__new__` does not withhold the exemption | 1 |
| validity asks the arrays alone, not what established them | 1 |
| restoration does not record what it established | 42 |
| a hook no rebuilding route calls makes the walk inconclusive | 1 |
| being an inexact subclass is disqualifying on its own | 22 |
| a name the copy protocol itself writes is read as a declaration | 13 |
| establishment is recorded against the problem, not the array | 171 |
| `__new__` is judged by how it is spelled | 1 |
| a `complex` number is no longer atomic | 1 |
| a reducer registered through `copyreg` is not noticed | 1 |
| any registration is excused for a modelled type | 1 |
| storage ownership is walked as though a copy rebuilt through it | 9 |

Several of these are detected only because their fixtures were changed, and
the change is the same each time. A guard reached by three mechanisms cannot
be witnessed by an instance that all three reach: the marker rules need a
subclass that overrides `f`, `F` and `G` *and* has had its buffer **deleted**,
so that reader identity sees nothing of the root's and presence has no name to
resolve, leaving the marker as the only thing able to speak. Redundant
coverage is not evidence for the rule it appears to cover, and every widening
added here narrowed what could witness the mechanisms beside it. Widening
presence from `ndarray` values to any resolving name did it again: fixtures
holding a `memoryview` in place of a coefficient had been proof against
presence and no longer were, so they now delete the name instead.

The ownership half of the refusal was initially **undetected**, and is
recorded for the reason [C-15.5](../../NUMERICS.md#c-15) already gives: once `__init__` and
`__setstate__` both establish ownership, no construction route reaches a
frozen non-owning buffer, so no behavioural test can exercise that branch. It
is now asserted directly, by building the state by hand, exactly as the
control-independence gate was.

Evidence: `tests/test_coefficient_immutability.py`.

What this does not reach, per [C-1](../../NUMERICS.md#c-1): a caller who builds an instance
through `object.__new__` and populates `__dict__` directly runs neither
`__init__` nor `__setstate__`, and a caller who holds a reference taken before
the buffer was frozen. The refusal above narrows the gap to those, rather than
closing it.

