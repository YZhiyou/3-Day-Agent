### 模块三：从文本到结构化数据 —— `PydanticOutputParser` 核心实战

在前面的模块中，模型输出的都是自然语言文本。但在实际应用中，我们常常需要将模型的回答解析为结构化的数据（如 JSON 对象），以便后续代码进行处理。例如，我们希望将用户的自然语言输入转化为一个包含“任务列表”、“优先级”等字段的 Python 字典或对象。

`PydanticOutputParser` 就是 LangChain 中专门用来解决这个问题的组件。它利用 **Pydantic** 库的数据校验能力，确保模型输出符合我们预定义的数据结构。

#### 1. 工作原理与数据流闭环

`PydanticOutputParser` 的工作流程构成了一个完美的数据闭环：

```
[用户输入] -> [Prompt模板 (包含格式说明)] -> [LLM] -> [LLM文本响应] -> [Parser解析与校验] -> [Pydantic对象实例]
```

具体步骤如下：
1. **定义数据模型**：我们先用 Pydantic 定义一个类，描述我们期望的数据结构。
2. **生成格式指令**：`PydanticOutputParser` 会根据这个 Pydantic 模型自动生成一段文本，详细描述 JSON 格式要求。这段文本会被注入到 Prompt 中，告诉 LLM 应该如何输出。
3. **LLM 生成**：模型接收到包含格式指令的 Prompt 后，会尝试返回一个符合要求的 JSON 字符串。
4. **解析与校验**：`PydanticOutputParser` 接收 LLM 的文本输出，将其解析成 JSON，然后用 Pydantic 模型进行校验。如果校验成功，返回一个 Pydantic 对象；如果失败，则抛出异常。

#### 2. 完整工作流代码示例

我们来构建一个“任务规划器”，将用户的一句话描述解析为一个结构化的任务列表。

**第一步：定义 Pydantic 数据模型**

```python
from pydantic import BaseModel, Field
from typing import List

class Task(BaseModel):
    """单个任务的数据结构"""
    title: str = Field(description="任务的简短标题")
    priority: int = Field(description="任务优先级，1为最高，5为最低", ge=1, le=5)
    estimated_hours: float = Field(description="预计完成所需小时数", ge=0)

class TaskPlan(BaseModel):
    """整体任务计划的数据结构"""
    tasks: List[Task] = Field(description="待办任务列表")
    overall_goal: str = Field(description="对用户输入目标的简洁总结")
```

这里我们使用了 `Field` 来为每个字段添加描述和约束（`ge` 表示大于等于，`le` 表示小于等于），这些信息会被 `PydanticOutputParser` 用来生成详细的格式说明，极大地提高了模型输出正确格式的概率。

**第二步：创建 Parser 并生成格式指令**

```python
from langchain_core.output_parsers import PydanticOutputParser

# 实例化解析器，并告诉它我们要解析成哪个 Pydantic 模型
parser = PydanticOutputParser(pydantic_object=TaskPlan)

# 获取格式指令字符串
format_instructions = parser.get_format_instructions()

# 打印看看格式指令长什么样
print("=== 自动生成的格式指令 ===")
print(format_instructions)
```

你会看到类似这样的输出（取决于 LangChain 版本）：
```
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:

{"properties": {"tasks": {"title": "Tasks", "description": "待办任务列表", "type": "array", "items": {"$ref": "#/definitions/Task"}}, "overall_goal": {"title": "Overall Goal", "description": "对用户输入目标的简洁总结", "type": "string"}}, "required": ["tasks", "overall_goal"], "definitions": {"Task": {"title": "Task", "description": "单个任务的数据结构", "type": "object", "properties": {"title": {"title": "Title", "description": "任务的简短标题", "type": "string"}, "priority": {"title": "Priority", "description": "任务优先级，1为最高，5为最低", "type": "integer", "minimum": 1, "maximum": 5}, "estimated_hours": {"title": "Estimated Hours", "description": "预计完成所需小时数", "type": "number", "minimum": 0}}, "required": ["title", "priority", "estimated_hours"]}}}
```

这段 JSON Schema 描述了对 LLM 输出的精确要求。LLM 被训练来遵循这类指令。

**第三步：构建包含格式指令的 Prompt**

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的任务规划助手。请根据用户的输入，生成一个详细的任务计划。\n{format_instructions}"),
    ("human", "我的目标是：{goal}")
])

# 将格式指令部分填入模板
prompt = prompt.partial(format_instructions=parser.get_format_instructions())
```

注意我们使用了 `.partial()` 方法，它可以将 `format_instructions` 这个固定的变量预先填充到模板中。这样后续调用时只需要传入 `goal` 变量即可。

**第四步：创建链并调用**

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)  # 低温度提高格式遵循度

# 构建 LCEL 链
chain = prompt | model | parser

# 调用链
user_goal = "我想在一周内学会使用 LangChain 的基础知识。"
try:
    result = chain.invoke({"goal": user_goal})
    
    print("\n=== 解析成功！===")
    print(f"总体目标: {result.overall_goal}")
    print("任务列表:")
    for i, task in enumerate(result.tasks, 1):
        print(f"  {i}. 任务: {task.title}, 优先级: {task.priority}, 预计耗时: {task.estimated_hours}小时")
        
    # 验证一下返回对象的类型
    print(f"\n返回对象类型: {type(result)}")
    
except Exception as e:
    print(f"解析失败: {e}")
```

**预期输出示例：**
```
=== 解析成功！===
总体目标: 一周内掌握 LangChain 基础知识
任务列表:
  1. 任务: 学习 LangChain 核心概念与架构, 优先级: 1, 预计耗时: 2.0小时
  2. 任务: 实践 Model 与 Prompt 组件的使用, 优先级: 1, 预计耗时: 3.0小时
  3. 任务: 掌握 PydanticOutputParser 处理结构化数据, 优先级: 2, 预计耗时: 2.5小时
  ...
返回对象类型: <class '__main__.TaskPlan'>
```

从输出可以看到，`result` 已经是一个强类型的 `TaskPlan` 对象，你可以安全地访问它的属性（如 `result.tasks[0].title`），并获得 IDE 的自动补全支持。

#### 3. 异常处理与容错机制

尽管 Pydantic 的 Schema 指令很强大，但 LLM 仍然可能偶尔“任性”地输出不符合要求的内容，例如：
- 在 JSON 前后添加解释性文字（例如："好的，这是您要的任务计划：```json {...}```"）。
- JSON 字符串中包含语法错误。

对于这些情况，LangChain 提供了 `OutputFixingParser` 和 `RetryOutputParser` 等容错解析器。这里介绍最常用的 `OutputFixingParser`，它会在首次解析失败后，将错误信息和原始输出一起发给 LLM，让它自行修正。

**示例：使用 `OutputFixingParser` 进行容错**

```python
from langchain.output_parsers import OutputFixingParser
from langchain_openai import ChatOpenAI

# 原始解析器
base_parser = PydanticOutputParser(pydantic_object=TaskPlan)

# 创建一个带修复功能的解析器，需要传入一个 LLM 实例用于修复
fixing_parser = OutputFixingParser.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0),
    parser=base_parser
)

# 模拟一个格式有问题的 LLM 输出（例如前后加了额外文字）
bad_response = """
好的，这是为您生成的任务计划：

{
  "tasks": [
    {"title": "阅读文档", "priority": 1, "estimated_hours": 2.0}
  ],
  "overall_goal": "学习 LangChain"
}

希望对您有帮助！
"""

print("=== 尝试用基础解析器 ===")
try:
    base_parser.parse(bad_response)
except Exception as e:
    print(f"基础解析器失败: {e}")

print("\n=== 使用修复解析器 ===")
try:
    fixed_result = fixing_parser.parse(bad_response)
    print("修复成功！解析出的对象：")
    print(fixed_result)
except Exception as e:
    print(f"修复解析器也失败了: {e}")
```

运行后你会发现，基础解析器会因为无法找到纯 JSON 而失败，但 `OutputFixingParser` 会将错误信息和原始文本一起发送给 LLM，LLM 能够理解问题所在，并返回一个干净的 JSON 字符串，最终解析成功。

#### 4. 模块小结

- `PydanticOutputParser` 是连接非结构化文本和结构化数据的关键桥梁。
- 通过定义 Pydantic 模型并注入自动生成的格式指令，可以显著提高 LLM 输出结构化数据的成功率。
- 在实际生产环境中，推荐配合 `OutputFixingParser` 使用，以增强系统的健壮性。
