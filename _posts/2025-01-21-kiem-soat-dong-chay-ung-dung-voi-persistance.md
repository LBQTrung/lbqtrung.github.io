---
layout: post
title: "Kiểm Soát Dòng Chảy Ứng Dụng với Persistence"
date: 2025-01-20
categories: [AI Agent, LangGraph]
tags: [AI, LangGraph, LangChain]
---

Bạn đã bao giờ tự hỏi làm thế nào để một ứng dụng LangChain có thể "ghi nhớ" trạng thái của nó, cho phép bạn can thiệp, chỉnh sửa, hoặc thậm chí quay ngược thời gian? Câu trả lời nằm ở cơ chế **persistence** mạnh mẽ của LangGraph, được thực hiện thông qua các **checkpointer**. Hãy cùng khám phá sức mạnh ẩn sau lớp "hậu trường" này!

## **Checkpointer và "Dòng Thời Gian" của Ứng Dụng**

LangGraph tích hợp sẵn một lớp persistence, cho phép lưu lại trạng thái của đồ thị ứng dụng tại mỗi bước (super-step). Các **checkpointer** chính là "người giữ lửa" của cơ chế này, tạo ra các "dấu mốc" thời gian cho phép bạn truy cập lại trạng thái của ứng dụng sau khi nó đã hoàn thành.

Mỗi "dấu mốc" này được gán một **thread**, một định danh duy nhất, cho phép bạn xem lại, chỉnh sửa, hoặc tiếp tục thực thi ứng dụng từ một thời điểm nhất định.

## **Threads - Định Danh cho Mỗi Phiên Làm Việc**

Khi gọi một đồ thị LangGraph có checkpointer, bạn cần chỉ định một thread_id trong phần cấu hình:

```python
{"configurable": {"thread_id": "1"}}
```

Mỗi thread_id đại diện cho một “dòng thời gian” riêng biệt, cho phép quản lý nhiều phiên làm việc (Ví dụ: Một ứng dụng chatbot có nhiều người cùng sử dụng thì cần xử lý nhiều cuộc trò chuyện khác nhau) một cách độc lập.

## **Checkpoints - "Ảnh Chụp" Trạng Thái Ứng Dụng**

Mỗi **checkpoint** là một bản "ảnh chụp" trạng thái của đồ thị tại một super-step, được biểu diễn bằng đối tượng StateSnapshot với các thuộc tính quan trọng:

- `config`: Cấu hình của checkpoint.
- `metadata`: Metadata liên quan đến checkpoint.
- `values`: Giá trị của các kênh trạng thái (state channels) tại thời điểm đó.
- `next`: Một tuple chứa tên các node sẽ được thực thi tiếp theo trong đồ thị.
- `tasks`: Một tuple các đối tượng PregelTask chứa thông tin về các task tiếp theo, bao gồm cả thông tin lỗi nếu bước đó đã được thực thi trước đó.

Hãy xem xét ví dụ sau, để hiểu rõ hơn về các checkpoint được lưu lại:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": ""}, config)
```

Sau khi chạy đồ thị, chúng ta sẽ thấy 4 checkpoint được lưu lại:

1. Checkpoint rỗng, với START là node tiếp theo.
2. Checkpoint với input {'foo': '', 'bar': []} và node_a là node tiếp theo.
3. Checkpoint với output của node_a là {'foo': 'a', 'bar': ['a']} và node_b là node tiếp theo.
4. Checkpoint với output của node_b là {'foo': 'b', 'bar': ['a', 'b']} và không có node tiếp theo.

Lưu ý rằng giá trị của bar chứa cả output của node_a và node_b, do chúng ta đã định nghĩa một reducer add cho field này.

## **Truy Cập Trạng Thái Đồ Thị**

Để tương tác với trạng thái đã lưu, bạn cần chỉ định thread_id.

- **Lấy Trạng Thái Mới Nhất:** Sử dụng `graph.get_state(config)` để lấy `StateSnapshot` của checkpoint mới nhất.
- **Lấy Trạng Thái Theo checkpoint_id:** Bạn có thể chỉ định `checkpoint_id` trong config để lấy trạng thái tại checkpoint cụ thể.

Code minh họa:

```python
# Lấy checkpoint mới nhất
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)

# Lấy checkpoint theo checkpoint_id
config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
graph.get_state(config)
```

## **Xem Lịch Sử Trạng Thái**

Sử dụng `graph.get_state_history(config)` để lấy toàn bộ lịch sử thực thi của đồ thị, bao gồm tất cả các `StateSnapshot` đã lưu cho một `thread_id` cụ thể. Các checkpoint sẽ được sắp xếp theo thứ tự thời gian, với checkpoint mới nhất ở đầu danh sách:

```python
config = {"configurable": {"thread_id": "1"}}
list(graph.get_state_history(config))
```

## **"Du Hành Thời Gian" với Replay**

Bạn có thể "quay lại" quá khứ bằng cách sử dụng `replay`. Khi gọi đồ thị với cả `thread_id` và `checkpoint_id`, LangGraph sẽ:

1. **Replay:** Thực thi lại các bước trước checkpoint được chỉ định.
2. **Thực Thi Tiếp:** Tiếp tục thực thi các bước sau checkpoint đó (tạo một nhánh mới, ngay cả khi chúng đã được thực thi trước đó).

```python
config = {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}
graph.invoke(None, config=config)
```

## **Memory Store - "Bộ Nhớ" Chung Giữa Các Thread**

Các checkpointer cho phép bạn quản lý trạng thái của ứng dụng trong từng `thread`, nhưng làm thế nào để chia sẻ thông tin giữa các `thread` khác nhau? Đây là lúc `Store` xuất hiện.

`Store` cho phép bạn lưu trữ thông tin chung, có thể truy cập từ bất kỳ `thread` nào, ví dụ như thông tin người dùng trong một chatbot.

Hãy xem cách sử dụng `InMemoryStore`:

```python
from langgraph.store.memory import InMemoryStore
import uuid
in_memory_store = InMemoryStore()

user_id = "1"
namespace_for_memory = (user_id, "memories")

memory_id = str(uuid.uuid4())
memory = {"food_preference" : "I like pizza"}
in_memory_store.put(namespace_for_memory, memory_id, memory)

memories = in_memory_store.search(namespace_for_memory)
memories[-1].dict()
```

Mỗi memory là một đối tượng `Item`, có các thuộc tính như `value`, `key`, `namespace`, `created_at`, và `updated_at`.

## **Semantic Search - Tìm Kiếm Thông Minh**

`Store` hỗ trợ tìm kiếm semantic, cho phép bạn tìm memory dựa trên ý nghĩa thay vì tìm kiếm chính xác. Để kích hoạt tính năng này, bạn cần cấu hình `Store` với một embedding model:

```python
from langchain.embeddings import init_embeddings

store = InMemoryStore(
  index={
      "embed": init_embeddings("openai:text-embedding-3-small"),
      "dims": 1536,
      "fields": ["food_preference", "$"]
  }
)
```

Sau đó, bạn có thể tìm kiếm bằng ngôn ngữ tự nhiên:

```python
memories = store.search(
  namespace_for_memory,
  query="What does the user like to eat?",
  limit=3
)
```

## Kết hợp Store và LangGraph

Để sử dụng Store trong LangGraph, bạn cần compile đồ thị với cả checkpointer và store:

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = graph.compile(checkpointer=checkpointer, store=in_memory_store)
```

Truy cập store trong các node bằng cách khai báo nó như một tham số trong hàm node:

```python
def update_memory(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())
    store.put(namespace, memory_id, {"memory": memory})

def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    memories = store.search(
        namespace,
        query=state["messages"][-1].content,
        limit=3
    )
    info = "\n".join([d.value["memory"] for d in memories])
```

**Lưu Ý:** Khi sử dụng LangGraph Platform, store mặc định sẽ có sẵn, bạn chỉ cần cấu hình indexing cho semantic search trong langgraph.json.

## Kết luận

Cơ chế persistence của LangGraph thông qua checkpointer là một công cụ mạnh mẽ, mang lại khả năng kiểm soát và độ tin cậy cao cho các ứng dụng của bạn. Với khả năng "du hành thời gian", quản lý "trí nhớ" và khôi phục lỗi, checkpointer mở ra nhiều khả năng mới cho các ứng dụng AI phức tạp.

Hy vọng bài viết này đã giúp bạn hiểu rõ hơn về sức mạnh của persistence trong LangGraph. Hãy khám phá và ứng dụng nó để tạo ra những trải nghiệm tuyệt vời!

Toàn bộ mã nguồn được triển khai tại: [https://colab.research.google.com/drive/1YS-8ylRbu1efYTYoDv3CD4wDKgzQS3y9?usp=sharing](https://colab.research.google.com/drive/1YS-8ylRbu1efYTYoDv3CD4wDKgzQS3y9?usp=sharing)

## Tài liệu tham khảo

- [https://langchain-ai.github.io/langgraph/concepts/low_level/#using-in-langgraph](https://langchain-ai.github.io/langgraph/concepts/low_level/#using-in-langgraph)
- [https://langchain-ai.github.io/langgraph/concepts/persistence/#langgraph.store.base.PutOp.index](https://langchain-ai.github.io/langgraph/concepts/persistence/#langgraph.store.base.PutOp.index)
