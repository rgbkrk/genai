import tiktoken

MAX_TOKENS = {
    "gpt-3.5-turbo-0301": 2048,
    "gpt-3.5-turbo": 2048,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
}


# Copied from https://platform.openai.com/docs/guides/chat/introduction on 3/17/2023
# Modified to support gpt-4 as a best guess
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    # TODO: Watch for when the new models are released and update this
    if model == "gpt-3.5-turbo" or model == "gpt-4" or model == "gpt-4-0314":
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""  # noqa: E501
        )


def trim_messages_to_fit_token_limit(messages, model="gpt-3.5-turbo-0301", max_tokens=None):
    """Reduce the number of messages until they are below the max token limit."""
    num_tokens = num_tokens_from_messages(messages, model=model)

    if max_tokens is None:
        max_tokens = MAX_TOKENS[model]

    while num_tokens > max_tokens:
        messages.pop(0)
        num_tokens = num_tokens_from_messages(messages, model=model)
    return messages
