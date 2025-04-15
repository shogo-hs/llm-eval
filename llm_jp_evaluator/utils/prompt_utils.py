"""
プロンプト処理ユーティリティモジュール - LLM Leaderboard再現用
"""
import os
import json
import random
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from copy import deepcopy

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


def get_system_message(system_message_intro: str, instruction: str) -> str:
    """
    システムメッセージを作成
    
    Args:
        system_message_intro: システムメッセージ前文
        instruction: 指示文
        
    Returns:
        str: 結合されたシステムメッセージ
    """
    system_message = ""
    system_message += system_message_intro
    system_message += "\n"
    system_message += instruction
    return system_message


def get_few_shot_messages(
    dataset_path: Union[str, Path], 
    num_few_shots: int
) -> List[Dict[str, str]]:
    """
    few-shotメッセージを取得
    
    Args:
        dataset_path: データセットのJSONファイルパス
        num_few_shots: few-shotの数
        
    Returns:
        List[Dict[str, str]]: few-shotメッセージのリスト（各メッセージは{"role": str, "content": str}の形式）
    """
    if not isinstance(dataset_path, Path):
        dataset_path = Path(dataset_path)
        
    # few-shot用のデータを取得するパスを構築
    # 本来はtrain/以下から取得するが、なければ現在のデータセット内から抽出
    train_path = dataset_path.parent.parent / "train" / dataset_path.name
    
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        samples = train_data.get("samples", [])
    else:
        # trainデータがない場合は同じデータセットから取得
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", [])
        
    if not samples:
        return []
    
    # ランダムにサンプルを選択（固定シードのランダム）
    random.seed(42)  # 再現性のために固定シード
    few_shot_indices = random.sample(range(min(len(samples), 20)), min(num_few_shots, len(samples)))
    few_shot_samples = [samples[i] for i in few_shot_indices]
    
    # メッセージリストに変換
    few_shot_messages = []
    for sample in few_shot_samples:
        few_shot_messages.append({"role": "user", "content": sample["input"]})
        few_shot_messages.append({"role": "assistant", "content": sample["output"]})
    
    return few_shot_messages


def apply_chat_template(
    messages: List[Dict[str, str]], 
    chat_template_path: Optional[str] = None
) -> str:
    """
    チャットテンプレートを適用
    
    Args:
        messages: メッセージリスト
        chat_template_path: チャットテンプレートファイルのパス（オプション）
        
    Returns:
        str: 整形されたプロンプト
    """
    # Jinja2がインストールされていない場合はシンプルな結合を行う
    if not JINJA2_AVAILABLE or not chat_template_path:
        return simple_template(messages)
        
    # テンプレートファイルが存在するか確認
    if not os.path.exists(chat_template_path):
        print(f"Warning: Chat template file {chat_template_path} not found. Using simple template.")
        return simple_template(messages)
    
    try:
        # テンプレートを読み込み
        with open(chat_template_path, "r", encoding="utf-8") as f:
            template_str = f.read()
        
        # テンプレート適用
        template = Template(template_str)
        prompt = template.render(messages=messages, add_generation_prompt=True)
        return prompt
    except Exception as e:
        print(f"Error applying chat template: {str(e)}")
        return simple_template(messages)


def simple_template(messages: List[Dict[str, str]]) -> str:
    """
    シンプルなテンプレート（jinja2がない場合のフォールバック）
    
    Args:
        messages: メッセージリスト
        
    Returns:
        str: 整形されたプロンプト
    """
    result = []
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "system":
            result.append(f"System: {content}")
        elif role == "user":
            result.append(f"User: {content}")
        elif role == "assistant":
            result.append(f"Assistant: {content}")
        else:
            result.append(f"{role}: {content}")
    
    # litellm用にフォーマット調整
    return "\n\n".join(result)


def format_prompts_for_litellm(
    dataset: Any, 
    task_name: Optional[str] = None, 
    message_intro: Optional[str] = None,
    num_few_shots: int = 0,
    chat_template_path: Optional[str] = None
) -> List[List[Dict[str, str]]]:
    """
    litellm用にプロンプトをフォーマット
    
    LLM Leaderboardのプロンプト形式に合わせてデータセットのサンプルをフォーマットし、
    litellmで使用できるメッセージリストに変換します。
    
    Args:
        dataset: データセットオブジェクト（loader.Datasetの期待）
        task_name: タスク名（例: "jsquad", "wiki_ner", "jmmlu"）
        message_intro: メッセージの前文（None時はタスクに応じたデフォルト値）
        num_few_shots: few-shotの数（デフォルト: 0）
        chat_template_path: チャットテンプレートファイルのパス（指定時はテンプレート適用）
        
    Returns:
        List[List[Dict[str, str]]]: litellm用のメッセージリストのリスト
        （各サンプルに対するメッセージのリストを含む）
    
    Note:
        - メッセージの前文はタスクに応じて自動的に選択されます（英語mmluの場合は英語、それ以外は日本語）
        - few-shotの数が指定されている場合は、データセットのfew_shotsプロパティから取得します
        - チャットテンプレートパスが指定されている場合は、Jinja2テンプレート処理を行います
    """
    prompts = []
    
    # デフォルトのメッセージ前文
    if message_intro is None:
        if task_name and "mmlu_en" in task_name:
            message_intro = "The following text provides instructions for a certain task."
        else:
            message_intro = "以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。リクエストを適切に完了するための回答を記述してください。"
    
    # few-shotメッセージの取得（もしパスがあれば）
    few_shot_messages = []
    if num_few_shots > 0 and hasattr(dataset, 'few_shots') and len(dataset.few_shots) > 0:
        # データセットにfew-shot例が含まれている場合はそれを使用
        for i in range(min(num_few_shots, len(dataset.few_shots))):
            few_shot_messages.append({
                "role": "user",
                "content": dataset.few_shots[i].input
            })
            few_shot_messages.append({
                "role": "assistant",
                "content": dataset.few_shots[i].output
            })
    
    # 各サンプルに対してプロンプトを作成
    for sample in dataset.samples:
        # メッセージリストの作成
        messages = []
        
        # システムメッセージの追加（あれば）
        if message_intro:
            instruction = "\n".join([message_intro, dataset.instruction])
            messages.append({"role": "system", "content": instruction})
        
        # few-shotメッセージの追加
        messages.extend(deepcopy(few_shot_messages))
        
        # ユーザーメッセージの追加
        messages.append({"role": "user", "content": sample.input})
        
        # もしチャットテンプレートを適用するなら
        if chat_template_path:
            formatted_prompt = apply_chat_template(messages, chat_template_path)
            # テンプレート適用後の場合は特殊形式に
            messages = [{"role": "user", "content": formatted_prompt}]
        
        prompts.append(messages)
    
    return prompts


def format_mmlu_prompt(
    input_text: str, 
    instruction: str, 
    message_intro: str = "The following text provides instructions for a certain task."
) -> List[Dict[str, str]]:
    """
    MMLU用のプロンプト整形
    
    Args:
        input_text: 入力テキスト
        instruction: 指示文
        message_intro: メッセージ前文
        
    Returns:
        List[Dict[str, str]]: メッセージリスト
    """
    system_message = get_system_message(message_intro, instruction)
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": input_text}
    ]
    
    return messages
