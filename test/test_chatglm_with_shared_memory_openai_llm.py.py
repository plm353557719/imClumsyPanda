import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import asyncio
from argparse import Namespace
from agent.chatglm_with_shared_memory_openai_llm import ChatglmWithSharedMemoryOpenaiLLM
from agent.chatglm_with_shared_memory_openai_llm.args import parser



async def dispatch(args: Namespace):
    args_dict = vars(args)

    # logger.info(f'Running in {args.mode} mode')
    if args.mode == 'demo':
        if not os.path.isfile(args.dialogue_path):
            raise FileNotFoundError(f'Invalid dialogue file path for demo mode: "{args.dialogue_path}"')

    chatglm_instance = ChatglmWithSharedMemoryOpenaiLLM(args_dict)

    # 使用代理链运行一些示例输入
    chatglm_instance.agent_chain.run(input="我跟露露聊了什么?")
    chatglm_instance.agent_chain.run(input="她开心吗?")
    chatglm_instance.agent_chain.run(input="她有表达意见吗?")
    chatglm_instance.agent_chain.run(input="根据历史对话总结下?")
    chatglm_instance.agent_chain.run(input="""可以拓展下吗?，比如写个小作文。
    大纲：游戏的美好回忆，触不可及的距离，不在乎得失
    主题：露露的陪伴无比珍贵
    背景：游戏，通话，当下
    开篇需要以游戏相识你挑逗的话语讲起
    """)


if __name__ == '__main__':
    args = None
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(dispatch(args))
