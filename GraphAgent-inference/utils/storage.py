from langchain.storage import InMemoryByteStore

mem_store = InMemoryByteStore()

def store_user_input(input, mem_store):
    mem_store.mset([("user_input", input)])
    return input