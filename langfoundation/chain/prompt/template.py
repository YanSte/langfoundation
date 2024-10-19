from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate

################################################################################
# Static Node prompts
# This file contains all the static prompts used in the nodes
################################################################################


################################################################################
# Indent
################################################################################

BASE_NEWLINE = "\n"
BASE_DOUBLE_NEWLINE_PROMPT = "\n\n"


################################################################################
# History
################################################################################


BASE_HISTORY_SECTION_PROMPT = "## Current Conversation History"

BASE_HISTORY_SYSTEM_PROMPT_TEMPLATE = SystemMessagePromptTemplate.from_template(BASE_HISTORY_SECTION_PROMPT)


################################################################################
# Humain
################################################################################

# Prompt
# ----

BASE_HUMAIN_QUERY_PROMPT = """## Query
```
{input}
```"""

# Prompt Template
# ----

BASE_HUMAIN_QUERY_PROMPT_TEMPLATE = HumanMessagePromptTemplate.from_template(BASE_DOUBLE_NEWLINE_PROMPT + BASE_HUMAIN_QUERY_PROMPT)

################################################################################
# Error
################################################################################

# Prompt
# ----

BASE_ERROR_TRY_NUMBER_PROMPT = "Tried number {retries}):\n ```{previous_output}```"

BASE_CHAIN_ERROR_SECTION_PROMPT = """## Error
Your previous attempts didn't align with the instructions or the required JSON object format.
Try again, but approaching it from a different angle. Think step by step.
Make sure your new answer is different from previous errors submitted, previous:
```
{previous_outputs}
```"""

# Prompt Template
# ----

BASE_ERROR_SYSTEM_PROMPT_TEMPLATE = SystemMessagePromptTemplate.from_template(BASE_CHAIN_ERROR_SECTION_PROMPT)

################################################
# Fields
################################################

# Query
# ---

BASE_USER_QUERY_FIELD_PROMPT = "The initial query from the user."

# Raisonning
# ---

BASE_SHORT_RAISONNING_FIELD_PROMPT = (
    "In short, a step by step detailed explanation of the reasoning process. (In Less than 35 words. Expects string)."  # noqa: E501
)


BASE_MINI_SHORT_RAISONNING_FIELD_PROMPT = (
    "In short, a step by step detailed explanation of the reasoning process. (In Less than 10 words. Expects string)."  # noqa: E501
)

# Response
# ---

BASE_RESPONSE_FIELD_PROMPT = "Detailed and informative response. (Expects string)"
BASE_QUESTION_FIELD_PROMPT = "Detailed and informative question. (Expects string)"
BASE_RESPONSE_USER_QUERY_FIELD_PROMPT = "Detailed and informative response to user query. (Expects string)"
