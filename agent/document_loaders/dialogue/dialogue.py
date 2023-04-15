import json
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age


class Dialogue:
    """
    要更加抽象地自动构建对话模型， 使用类和方法来表示不同的对话元素。以下是一个基本的对话模型构建框架，您可以根据实际需求进行扩展和优化。
    首先，定义表示对话中不同元素的类：
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.turns = []

    def add_turn(self, turn):
        """
        创建对话参与者的实例
        :param turn:
        :return:
        """
        self.turns.append(turn)

    def display(self):
        for turn in self.turns:
            print(f"{turn.speaker.name}: {turn.message}")

    def export_to_file(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            for turn in self.turns:
                file.write(f"{turn.speaker.name}: {turn.message}\n")

    def to_dict(self):
        dialogue_dict = {"turns": []}
        for turn in self.turns:
            turn_dict = {
                "speaker": turn.speaker.name,
                "message": turn.message
            }
            dialogue_dict["turns"].append(turn_dict)
        return dialogue_dict

    def to_json(self):
        dialogue_dict = self.to_dict()
        return json.dumps(dialogue_dict, ensure_ascii=False, indent=2)

    def participants_to_export(self):
        """
        导出参与者
        :return:
        """
        participants = set()
        for turn in self.turns:
            participants.add(turn.speaker.name)
        return ', '.join(participants)


class Turn:
    def __init__(self, speaker, message):
        self.speaker = speaker
        self.message = message


class DialogueLoader(BaseLoader):
    """Load dialogue."""

    def __init__(self, dialogue: Dialogue):
        """Initialize with dialogue."""
        self.dialogue = dialogue

    def load(self) -> List[Document]:
        """Load from dialogue."""
        documents = []
        participants = self.dialogue.participants_to_export()

        for turn in self.dialogue.turns:
            metadata = {"source": f"对话文件：{self.dialogue.file_path},发起人：{turn.speaker.name}，参与人：{participants}"}
            turn_document = Document(page_content=turn.message, metadata=metadata.copy())
            documents.append(turn_document)

        return documents