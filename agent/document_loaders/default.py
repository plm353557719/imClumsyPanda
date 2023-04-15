from .dialogue.dialogue import *


def parse_dialogue(file_path):
    """
    这个 parse_dialogue 函数会读取指定的对话文件，然后逐行解析每个对话轮次。对于每个轮次，函数会从文本中提取发言人的名字和消息内容，
    并创建一个 Turn 实例。如果发言人还没有在参与者字典（participants）中，
    那么就创建一个新的 Person 实例。最后，将解析好的 Turn 实例添加到 Dialogue 对象中。
    请注意，这个示例代码假设文件中的每行都遵循特定的格式，即 <发言人>: <消息>。如果您的文件有不同的格式或包含其他元数据，您可能需要相应地调整解析逻辑。
    """
    dialogue = Dialogue(file_path)
    participants = {}
    speaker_name = None
    message = None

    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            if speaker_name is None:
                speaker_name, _ = line.split(':', 1)
            elif message is None:
                message = line
                if speaker_name not in participants:
                    participants[speaker_name] = Person(speaker_name, None)

                speaker = participants[speaker_name]
                turn = Turn(speaker, message)
                dialogue.add_turn(turn)

                # Reset speaker_name and message for the next turn
                speaker_name = None
                message = None

    return dialogue