def build_judge_markdown(
    initial_text: str,
    user_message: str,
    label_response: str,
    llm_response: str,
    user_context: str = "",
    preview_report_user_example: str = "",
    default_context: str = "",
) -> str:
    parts = []
    parts.append("<INITIAL-TEXT>\n" + (initial_text or "") + "\n</INITIAL-TEXT>")
    parts.append("<USER-MESSAGE>\n" + (user_message or "") + "\n</USER-MESSAGE>")
    parts.append("<LABEL-RESPONSE>\n" + (label_response or "") + "\n</LABEL-RESPONSE>")
    parts.append("<LLM-RESPONSE>\n" + (llm_response or "") + "\n</LLM-RESPONSE>")

    if user_context:
        parts.append("<USER-CONTEXT>\n" + user_context + "\n</USER-CONTEXT>")
    if preview_report_user_example:
        parts.append("<PREVIEW-REPORT-USER-EXAMPLE>\n" + preview_report_user_example + "\n</PREVIEW-REPORT-USER-EXAMPLE>")
    if default_context:
        parts.append("<DEFAULT-CONTEXT>\n" + default_context + "\n</DEFAULT-CONTEXT>")

    return "\n\n".join(parts) + "\n"
