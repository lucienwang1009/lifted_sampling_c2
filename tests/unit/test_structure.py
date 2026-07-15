from c2_wms.structure import PredicateKey, SampledStructure


def test_structure_has_stable_relation_and_atom_views():
    structure = SampledStructure.from_mapping(
        ("a", "b"),
        {
            PredicateKey("P", 1): {("b",), ("a",)},
            PredicateKey("R", 2): {("a", "b")},
        },
    )

    assert structure.relation("P", 1) == frozenset({("a",), ("b",)})
    assert structure.true_atoms() == ("P(a)", "P(b)", "R(a, b)")
