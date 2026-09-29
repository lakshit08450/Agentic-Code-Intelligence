from codeintel.stage2.compare import outputs_match


def test_whitespace_and_crlf_insensitive():
    assert outputs_match("1 2\r\n3\r\n", "1 2\n3")
    assert outputs_match("  1\n\n2  ", "1 2\n")


def test_lengths_must_match():
    assert not outputs_match("1 2", "1 2 3")
    assert not outputs_match("", "0")


def test_float_tolerance():
    assert outputs_match("0.3333333", "0.333333333")
    assert outputs_match("1e-7", "0")
    assert not outputs_match("0.34", "0.33")


def test_integers_exact():
    assert not outputs_match("1000000000000000001", "1000000000000000000")
    assert outputs_match("+5", "5")


def test_case_sensitivity_optional():
    assert not outputs_match("yes", "YES")
    assert outputs_match("yes", "YES", case_insensitive=True)


def test_huge_integers_do_not_crash():
    # regression: int() raised ValueError on a 101,110-digit token and killed the Stage 2 run
    big = "7" * 101_110
    assert outputs_match(big, big)
    assert not outputs_match(big, "7" * 101_109 + "8")
    assert outputs_match("-0", "0") and outputs_match("007", "7") and not outputs_match("-5", "5")
