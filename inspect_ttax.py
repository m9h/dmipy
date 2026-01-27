import ttax
print("ttax dir:", dir(ttax))
if hasattr(ttax, 'ops'):
    print("ttax.ops dir:", dir(ttax.ops))
if hasattr(ttax, 'decompositions'):
    print("ttax.decompositions dir:", dir(ttax.decompositions))

