import os
from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.client import User
from diagrams.programming.framework import Fastapi
from diagrams.programming.language import Python
from diagrams.onprem.database import PostgreSQL
from diagrams.generic.database import SQL
from diagrams.onprem.network import Nginx
from diagrams.generic.compute import Rack

def create_diagram():
    graph_attr = {
        "fontsize": "26",
        "fontname": "Helvetica-Bold",
        "pad": "1.2",
        "nodesep": "1.0",
        "ranksep": "1.4",
        "splines": "spline",
        "bgcolor": "white",
        "ratio": "0.7"
    }

    node_attr = {
        "fontsize": "12",
        "fontname": "Helvetica"
    }

    edge_attr = {
        "color": "#4A4A4A"
    }

    with Diagram(
        "Agentic Hierarchical RAG Architecture",
        show=False,
        filename="system_architecture",
        direction="LR",
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr
    ):

        # ================= USER =================
        user = User("User")

        # ================= INTERFACE =================
        with Cluster("Application Layer"):
            ui = Python("Streamlit UI")
            api = Fastapi("API Layer")

        # ================= CONTROL =================
        with Cluster("Control Plane"):
            orchestrator = Rack("Orchestrator")
            planner = Rack("Query Planner")
            controller = Rack("Retrieval Controller")

        # ================= RETRIEVAL =================
        with Cluster("Retrieval Layer"):
            hybrid = Rack("Hybrid Search")

            with Cluster("Retrievers"):
                dense = Rack("Dense (FAISS)")
                sparse = Rack("Sparse (BM25)")

            reranker = Rack("Reranker")

        # ================= REASONING =================
        with Cluster("Reasoning Layer"):
            compressor = Rack("Context Compression")
            graph_reasoner = Rack("Graph Reasoning")
            multi_pass = Rack("Multi-pass LLM")
            validator = Rack("Self-RAG Validator")

        # ================= DATA =================
        with Cluster("Data Layer"):
            vector_db = SQL("Vector DB")
            doc_store = PostgreSQL("Metadata Store")

        # ================= LLM =================
        llm = Rack("LLM (Qwen)")

        # ================= FLOW =================

        # User → Interface
        user >> ui
        user >> api

        # Interface → Control
        ui >> orchestrator
        api >> orchestrator

        orchestrator >> planner >> controller

        # Retrieval Flow
        controller >> Edge(label="Query Strategy") >> hybrid
        hybrid >> dense
        hybrid >> sparse

        dense >> vector_db
        sparse >> doc_store

        [dense, sparse] >> reranker

        # Context → Reasoning
        reranker >> Edge(label="Top-K Context") >> compressor

        compressor >> graph_reasoner >> multi_pass >> validator

        # Feedback Loop (cleaner)
        validator >> Edge(style="dashed", label="Refinement Loop") >> compressor

        # LLM interactions (subtle, consistent)
        controller >> Edge(style="dotted") >> llm
        multi_pass >> Edge(style="dotted") >> llm
        validator >> Edge(style="dotted") >> llm

        # Final Output
        validator >> Edge(label="Final Response") >> orchestrator


if __name__ == "__main__":
    create_diagram()
    print("Diagram generated successfully as system_architecture.png")
