---
layout: post
title: "LangGraph Project 3: AI Agent cơ bản"
date: 2025-02-12
categories: [LangGraph, Project]
tags: [AI, LangGraph, LangChain, Gemini]
---

Hôm nay mình sẽ chia sẻ với các bạn một dự án thú vị mang tên **"Dự án 3: Xây dựng AI Agent đơn giản"**. Trong bài viết này, chúng ta sẽ cùng nhau tìm hiểu cách tạo ra một Agent trí tuệ nhân tạo (AI Agent) có khả năng gọi tool bên ngoài bằng cách sử dụng một số thư viện mã nguồn mở như **LangChain**, **LangGraph** và **Google Generative AI**. Hãy cùng khám phá từng bước trong quá trình xây dựng và chạy workflow của dự án nhé!

---

## 1. Cài đặt thư viện 🚀

Đầu tiên, chúng ta cần cài đặt một số thư viện cần thiết. Đoạn code dưới đây giúp cài đặt các package như `langchain`, `langchain_core`, `langchain_community`, `langchain_google_genai` và `langgraph` một cách nhanh chóng:

```bash
!pip install langchain langchain_core langchain_community -q
!pip install langchain_google_genai -q
!pip install langgraph -q
```

Việc cài đặt này đảm bảo môi trường của bạn có đầy đủ các công cụ để triển khai Agent và làm việc với các mô hình ngôn ngữ tiên tiến.

## 2. Thiết Lập Dự Án 🔧

Tiếp theo, chúng ta thiết lập dự án bằng cách cấu hình các API key cần thiết. Ở đây, bạn sẽ cần key từ **Google API** và **Tavily API** (để thực hiện tìm kiếm trên web). Mình sử dụng `getpass` để nhập key một cách bảo mật:

```python
import os
import getpass

if "GOOGLE_API_KEY" not in os.environ:
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")

if "TAVILY_API_KEY" not in os.environ:
  os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Google API Key: ")

```

Bạn có thể truy cập vào [đây](https://app.tavily.com/home?code=SAWcRNUWjLA-9FLWcohdGmAp4xOfPJu4lvIEmevo6px0e&state=eyJyZXR1cm5UbyI6Ii9ob21lIn0) để lấy API key cho Tavily. Đối với Google API key thì các bạn vào Google AI Studio nha (Cái này đã được giới thiệu trong các dự án trước rồi nè)

## 3. Khởi Tạo LLM với Google Generative AI 🌟

Trong phần này, chúng ta sử dụng thư viện `langchain_google_genai` để khởi tạo một mô hình LLM (Large Language Model) từ Google. Hiện nay, Google đã phát hành Gemini 2.0 với nhiều tính năng mới. Cấu hình mô hình như sau:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
print("Initialized LLM successfully!")
```

Sau đó, chúng ta kết hợp mô hình này với công cụ tìm kiếm của Tavily:

```python
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]
model_with_tool = llm.bind_tools(tools)

```

Việc bind tools giúp mô hình không chỉ trả lời dựa trên dữ liệu huấn luyện mà còn có thể truy vấn dữ liệu mới từ internet khi cần thiết.

## 4. Định Nghĩa State cho Agent 🗂

Để theo dõi trạng thái của Agent trong quá trình tương tác, chúng ta định nghĩa một kiểu dữ liệu `AgentState`. Kiểu dữ liệu này chứa danh sách các message đã được trao đổi:

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], operator.add]
```

## 5. Khởi Tạo Các Node của Workflow 🔄

Chính nhờ LangGraph, chúng ta có thể xây dựng một workflow rõ ràng với các node thực hiện các bước khác nhau trong quá trình xử lý yêu cầu của người dùng. Ở đây, có hai node chính:

1. **Tool Node**: Node này sử dụng `TavilySearchResults` để truy xuất dữ liệu.
2. **Tool Invocation**: Node gọi mô hình LLM với trạng thái hiện tại và nhận kết quả phản hồi.

Đoạn code dưới đây định nghĩa hàm kiểm tra xem liệu workflow có cần tiếp tục hay dừng lại:

```python
def should_continue(state):
  messages = state["messages"]
  last_message = messages[-1]

  if last_message.tool_calls:
    return "continue"
  else:
    return "end"
```

Và hàm gọi mô hình:

```python
def call_model(state):
  messages = state["messages"]
  response = model_with_tool.invoke(messages)
  return {"messages": [response]}
```

## 6. Định Nghĩa WorkFlow của Agent 🕸

Bây giờ, chúng ta sẽ định nghĩa workflow cho Agent bằng cách sử dụng **StateGraph** từ LangGraph. Workflow này cho phép chuyển đổi linh hoạt giữa các node dựa trên kết quả trả về từ LLM và các tool:

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)
workflow.add_edge("action", "agent")

app = workflow.compile()
```

Qua đó, luồng xử lý được hình thành theo logic:

**Human → LLM → Tool → LLM → Response**

Ngoài ra, chúng ta còn trực quan hóa workflow thông qua biểu đồ Mermaid để có cái nhìn tổng quát về quá trình:

```python
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image

display(
    Image(app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API
    ))
)

```

## 7. Thử Nghiệm WorkFlow và Hiển Thị Quy Trình 🚀

Cuối cùng, chúng ta thực hiện thử nghiệm workflow bằng cách gửi một câu hỏi từ người dùng. Ví dụ, câu hỏi "What is the weather in Hue city?" sẽ được gửi qua hàm `call_app` và kết quả trả về được in ra:

```python
from langchain_core.messages import HumanMessage

def call_app(content):
  inputs = {"messages": [HumanMessage(content=content)]}
  results = app.invoke(inputs)
  return results["messages"][-1].content, results["messages"]

last_response, messages = call_app("What is the weather in Hue city?")
print(last_response)

```

---

## Kết Luận 💡

Trong bài viết này, chúng ta đã cùng nhau xây dựng một AI Agent đơn giản sử dụng các công cụ mạnh mẽ từ LangChain và LangGraph. Qua đó, ta không chỉ học cách khởi tạo và cấu hình mô hình LLM từ Google mà còn nắm bắt được quy trình xây dựng một workflow linh hoạt, có khả năng mở rộng theo các nhu cầu cụ thể.

Việc sử dụng workflow theo dạng **Human → LLM → Tool → LLM → Response** giúp ta tạo ra các ứng dụng AI có khả năng tương tác tự nhiên và hiệu quả, đồng thời dễ dàng tích hợp thêm nhiều công cụ khác nhau để mở rộng chức năng. Toàn bộ mã nguồn được triển khai tại: [https://colab.research.google.com/drive/13M7D2Vqa469Vybf_7HGu2ehwOuFlvBD1?usp=sharing](https://colab.research.google.com/drive/13M7D2Vqa469Vybf_7HGu2ehwOuFlvBD1?usp=sharing)

---

🌟 Nếu bạn thấy bài viết này hữu ích, đừng quên để lại một ⭐ trên repo của tác giả nhé!

**Tác giả:** Trung Lê

📧 **Email:** lebaquoctrung@gmail.com

💻 **GitHub:** [LBQTrung](https://github.com/LBQTrung)

🌐 **Website:** [lbqtrung.github.io](https://lbqtrung.github.io/)
