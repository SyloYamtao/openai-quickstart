import gradio as gr
import argparse

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

SALES_BOTS = {}
SALES_TYPE = "real_estate_sales"


# 初始化矢量数据库
def init_vector_store(file_path_and_name: str, vector_store_dir: str):
    # 读取数据集文件
    with open(file_path_and_name) as f:
        file_read = f.read()
    # 将数据集文件转换为documents对象
    docs = text_splitter.create_documents([file_read])
    # 初始化对应数据集文件的矢量数据库
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    # 矢量数据库数据持久化
    db.save_local( vector_store_dir)
    print(vector_store_dir + " has loaded in faiss successfully...")


# 初始化对应的销售机器人
def initialize_sales_bot(vector_store: str):
    # 如果对应数据集的.faiss文件存在,请将此行代码放开执行成功至少一次
    init_vector_store("data_set/" + vector_store + "_data.txt", "vector_store/" + vector_store)
    db = FAISS.load_local("vector_store/" + vector_store, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    SALES_BOTS[vector_store] = RetrievalQA.from_chain_type(llm,
                                                           retriever=db.as_retriever(
                                                               search_type="similarity_score_threshold",
                                                               search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOTS[vector_store].return_source_documents = True
    return SALES_BOTS[vector_store]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# 获得命令行启动参数
def get_arguments_enable_chat():
    # 从命令行参数中获取
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_chat', type=str2bool, default=False)
    args = parser.parse_args()
    return args.enable_chat


ENABLE_CHAT = get_arguments_enable_chat()


# 聊天检索
def sales_chat(sales_type, message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    print(f"[SALES_TYPE]{sales_type}")
    print(f"[ENABLE_CHAT]{ENABLE_CHAT}")
    ans = SALES_BOTS[sales_type]({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or ENABLE_CHAT:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"


def launch_gradio():
    sales_type = gr.Dropdown(
        choices=[
            ('房地产销售顾问', "real_estate_sales"),
            ("电器销售顾问", "electrical_appliance_sales"),
            ("家装销售顾问", "home_decoration_sales"),
            ("教育销售顾问", "education_sales")
        ],
        value="real_estate_sales",
        label="房地产销售顾问类型",
        info="选择销售顾问的类型(默认为房地产销售顾问)"
    )

    def wrapper_fn(message, history, sales_type):
        return sales_chat(sales_type, message, history)

    chat_interface = gr.ChatInterface(
        additional_inputs=[sales_type],
        fn=wrapper_fn,
        title="销售聊天机器人",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )
    chat_interface.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot("real_estate_sales")
    # 初始化电器销售机器人
    initialize_sales_bot("electrical_appliance_sales")
    # 初始化家装销售机器人
    initialize_sales_bot("home_decoration_sales")
    # 初始化教育销售机器人
    initialize_sales_bot("education_sales")
    # 启动 Gradio 服务
    launch_gradio()
