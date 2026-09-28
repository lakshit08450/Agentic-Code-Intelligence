from codeintel.stage2.sampleio import parse

CF = """Find it.

-----Input-----

One line with a and b.

-----Output-----

Print the answer.

-----Examples-----
Input
19 29

Output
2

Input
3 6

Output
3



-----Note-----

The first example: ...
"""

CF_SINGLE_COLON = "Stuff.\n\n-----Example-----\nInput:\n2\n2 3\n\nOutput:\n8\n89\n\n-----Explanation-----\nCase 1.\n"

CODECHEF = "-----Input:-----\n- The first line has T.\n\n-----Output:-----\nPrint.\n\n-----Sample Input:-----\n1\ncodechef\n\n-----Sample Output:-----\n173\n\n-----EXPLANATION:-----\nBecause.\n"

EXAMPLE_IO = "-----Example Input-----\n1\n..P.P\n\n-----Example Output-----\nYes\n\n-----Explanation-----\nx"

ATCODER_TYPO = "-----Sample Input-----\n1\n3 2\n1 2 3\n\n-----Example Ouput-----\n3 1 0\n\n-----Explanation-----\n..."

ATCODER_NUMBERED = (
    "-----Sample Input 1-----\n3\n\n-----Sample Output 1-----\n6\n\n"
    "-----Sample Input 2-----\n4\n\n-----Sample Output 2-----\n24\n"
)

LEETCODE = "Given nums.\n\nExample 1:\nInput: nums = [1,2,3], k = 3\nOutput: 12\nExplanation: ...\n\nConstraints:\n1 <= n\n"

CODEWARS = "# Example\n\n For `n = 3, k = 2`, the result should be `\"ODD\"`\n\n# Input/Output\n - `[input]` integer `n`\n"


def test_codeforces_examples_block():
    s = parse(CF)
    assert s.kind == "stdin"
    assert s.pairs == [("19 29\n", "2\n"), ("3 6\n", "3\n")]


def test_input_spec_is_not_sample():
    assert all("One line" not in i for i, _ in parse(CF).pairs)


def test_example_block_with_colons():
    assert parse(CF_SINGLE_COLON).pairs == [("2\n2 3\n", "8\n89\n")]


def test_codechef_sample_sections():
    assert parse(CODECHEF).pairs == [("1\ncodechef\n", "173\n")]


def test_example_input_output_sections():
    assert parse(EXAMPLE_IO).pairs == [("1\n..P.P\n", "Yes\n")]


def test_ouput_typo():
    assert parse(ATCODER_TYPO).pairs == [("1\n3 2\n1 2 3\n", "3 1 0\n")]


def test_numbered_samples():
    assert parse(ATCODER_NUMBERED).pairs == [("3\n", "6\n"), ("4\n", "24\n")]


def test_call_based():
    assert parse(LEETCODE).kind == "call_based"
    assert parse(CODEWARS).kind == "call_based"


def test_no_samples():
    assert parse("Just a statement with no examples.").kind == "none"


def test_mismatch_is_parse_failure():
    assert parse("-----Sample Input-----\n1\n").kind == "parse_failure"
