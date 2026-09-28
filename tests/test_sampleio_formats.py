"""One test per sample format / fix added after the gold-failure review (docs/LOG.md, step 4)."""

from codeintel.stage2.sampleio import drop_blank_lines, parse, trim_output


def test_inline_data_in_dashed_example_block():
    st = "-----Input-----\nspec\n\n-----Example-----\nInput:4\n5 6\n6 10\nOutput:3.00000 4.00000 5.00000\n-1\n"
    assert parse(st).pairs == [("4\n5 6\n6 10\n", "3.00000 4.00000 5.00000\n-1\n")]


def test_inline_data_not_used_in_plain_leetcode_blocks():
    st = "Example 1:\nInput: nums = [1,2,3], k = 3\nOutput: 12\n"
    assert parse(st).kind == "call_based"


def test_example_text_case_title():
    st = "-----Example Text Case-----\nInput:\n\n1\n10 6\n8 8 3 5 3 8 5 7 7 7\n\nOutput:\n37\n"
    assert parse(st).pairs == [("1\n10 6\n8 8 3 5 3 8 5 7 7 7\n", "37\n")]


def test_header_with_trailing_text():
    st = "-----EXAMPLE-----Input:\n\n1\n\n7\n\n20 6 5\nOutput:\n\n1: 2\n\n3: 2\n"
    s = parse(st)
    assert s.pairs == [("1\n7\n20 6 5\n", "1: 2\n\n3: 2\n")]


def test_duplicated_header_text_dropped():
    st = "-----Sample Input:-----Sample Input:\n2\n3\n-----Sample Output:-----Sample Output:\n6\n"
    assert parse(st).pairs == [("2\n3\n", "6\n")]


def test_glued_example_input():
    st = "-----ExampleInput:-----\n3\n-----ExampleOutput:-----\n9\n"
    assert parse(st).pairs == [("3\n", "9\n")]


def test_sample_input_followed_by_generic_output_header():
    st = "-----Input-----\nspec\n-----Output-----\nspec\n-----Sample Input-----\n4\n-----Output-----\n16\n"
    assert parse(st).pairs == [("4\n", "16\n")]


def test_equals_headers():
    st = "=====Sample Input=====\n1 2\n=====Sample Output=====\n3\n"
    assert parse(st).pairs == [("1 2\n", "3\n")]


def test_trim_explanation_after_blank_line():
    assert trim_output("2\n\nThe second rectangle should be moved to the left by a distance of 2.\n") == "2\n"
    assert trim_output("4\n\nThere are four ways to divide the people:\n - (1,2),(3)\n") == "4\n"
    assert trim_output("23\n\n(Explanation: 10+3+7+3)\n") == "23\n"
    assert trim_output("1 1 0\nExplanation\nTotal of 3 particles\n") == "1 1 0\n"
    assert trim_output("5\n\nNote: Your program should not print anything else.\n") == "5\n"


def test_trim_keeps_blank_separated_answers():
    assert trim_output("200000002\n\n0\n") == "200000002\n\n0\n"
    assert trim_output("YES\n\nNO\n") == "YES\n\nNO\n"
    assert trim_output("Chef wins\n") == "Chef wins\n"


def test_blank_lines_dropped_from_inputs():
    assert drop_blank_lines("2\n\n5\n\n72  \n") == "2\n5\n72\n"
    st = "-----Sample Input:-----\n2\n\n5\n\n72  \n\n-----Sample Output:-----\n3\n\n12  \n\n-----EXPLANATION:-----\nx"
    assert parse(st).pairs == [("2\n5\n72\n", "3\n\n12\n")]
