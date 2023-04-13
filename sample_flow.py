import prefect
from prefect import flow

@flow
def hello_world():
    print("Hello from Prefect!")


hello_world()