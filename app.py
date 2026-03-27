import gradio as gr
from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import SiliconflowChat
from RAG.Embeddings import BgeWithAPIEmbedding

#输入文本处理程序
def greet(name):
  # 没有保存数据库
  docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) # 获得data目录下的所有文件内容并分割
  vector = VectorStore(docs)
  embedding = BgeWithAPIEmbedding() # 创建EmbeddingModel
  vector.get_vector(EmbeddingModel=embedding)
  vector.persist(path='storage') # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

  vector.load_vector('./storage') # 加载本地的数据

  content = vector.query(name, EmbeddingModel=embedding, k=1)[0]
  chat = SiliconflowChat()
  return chat.chat(name, [], content)

#接口创建函数
#fn设置处理函数，inputs设置输入接口组件，outputs设置输出接口组件
#fn,inputs,outputs都是必填函数
demo = gr.Interface(fn=greet, inputs=gr.Textbox(
        lines=5,
        autofocus=True,
        placeholder='Qwen/Qwen2.5-7B-Instruct'
    ), outputs=gr.Textbox(
        lines=5
    ), 
    examples=[
        "你好",
        "自我介绍",
        "基础术语",],
    title="测试RAG")
demo.launch()