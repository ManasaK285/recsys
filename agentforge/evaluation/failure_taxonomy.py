FAILURE_CLASSES = {
    'planning.incomplete_decomposition','retrieval.wrong_file','retrieval.insufficient_context',
    'implementation.syntax_error','implementation.logic_error','implementation.api_misuse',
    'verification.missing_test','verification.test_failure','workflow.bad_tool_selection',
    'workflow.excessive_retries','workflow.bad_recovery_strategy'
}

def classify(verification: dict) -> str:
    if verification.get('tests_failed',0): return 'verification.test_failure'
    if verification.get('lint') != 'PASS': return 'implementation.syntax_error'
    if verification.get('typecheck') != 'PASS': return 'implementation.api_misuse'
    return 'workflow.bad_recovery_strategy'
