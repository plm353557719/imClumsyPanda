import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
import asyncio
from argparse import Namespace
from agent.chatglm_with_shared_memory_openai_llm.args import parser
from agent.chatglm_with_shared_memory_openai_llm.chatglm_with_shared_memory_openai_llm import *


async def dispatch(args: Namespace):
    args_dict = vars(args)

    # logger.info(f'Running in {args.mode} mode')
    if args.mode == 'demo':
        if not os.path.isfile(args.dialogue_path):
            raise FileNotFoundError(f'Invalid dialogue file path for demo mode: "{args.dialogue_path}"')

    chatglm_instance = ChatglmWithSharedMemoryOpenaiLLM(args_dict)

    # 使用代理链运行一些示例输入
    chatglm_instance.agent_chain.run(input="你好，帮我看下我之前哪句话不合适")




if __name__ == '__main__':
    args = None
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(dispatch(args))

