import re



model_output = '**Final Answer:**\n53049077962937137584 o1-mini False'
answer_match = re.search(r'Final Answer.*?\s+(\S+)', model_output, re.M)
print (answer_match.group(1))
