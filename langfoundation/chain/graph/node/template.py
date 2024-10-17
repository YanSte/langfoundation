from langchain_core.prompts.chat import HumanMessagePromptTemplate

BASE_THINK_STEP_BY_STEP_PROMPT = """
Let’s Think, Step by Step.
Take your time to explain the reasoning behind every step to ensure a thorough understanding.
""".strip()


BASE_THINK_STEP_BY_STEP_HUMAIM_PROMPT_TEMPLATE = HumanMessagePromptTemplate.from_template(BASE_THINK_STEP_BY_STEP_PROMPT)


BASE_EMTPY_MSG_HUMAIM_PROMPT_TEMPLATE = HumanMessagePromptTemplate.from_template("\n\n")
