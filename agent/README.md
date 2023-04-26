### 对话记忆和聊天记录分析行为

> 基于openai的任务处理，实现本地文档调用链的示例，使用chatglm处理一部分任务事件，使用chatglm来完成记忆共享（SharedMemory）

```bash
#执行之前需要检查环境这几个环境变量
$ export OPENAI_API_KEY=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
$ export http_proxy=http://192.168.3.181:7890
$ export https_proxy=http://192.168.3.181:7890
# 启动
$ python -m agent.chatglm_with_shared_memory_openai_llm  --dialogue-path=/media/gpt4-pdf-chatbot-langchain/langchain-ChatGLM/content/state_of_the_history.txt

```

