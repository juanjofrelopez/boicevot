from pgvector.psycopg import register_vector
import psycopg
import time


class PgVectorConnection:
    def __init__(self, migrate_mode=False):
        self.conn = psycopg.connect(
            "host=localhost port=5432  dbname=postgres user=postgres password=postgres",
            autocommit=True,
        )
        self.initialize(migrate_mode)

    def initialize(self, migrate_mode):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self.conn)
            if migrate_mode:
                cur.execute("DROP TABLE IF EXISTS documents")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS documents (id bigserial PRIMARY KEY, content text, embedding vector(384))"
            )
            print("Connected to db successfully!")

    def close(self):
        self.conn.close()

    def save_embeddings(self, embeddings):
        for content, embedding in embeddings:
            self.conn.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                (content, embedding),
            )

    def find_similar(self, query, embedding, k=60):
        start_time = time.time()
        sql = """
            WITH semantic_search AS (
                SELECT id, RANK () OVER (ORDER BY embedding <=> %(embedding)s) AS rank
                FROM documents
                ORDER BY embedding <=> %(embedding)s
                LIMIT 20
            ),
            keyword_search AS (
                SELECT id, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC)
                FROM documents, plainto_tsquery('english', %(query)s) query
                WHERE to_tsvector('english', content) @@ query
                ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC
                LIMIT 20
            )
            SELECT
                COALESCE(semantic_search.id, keyword_search.id) AS id,
                COALESCE(1.0 / (%(k)s + semantic_search.rank), 0.0) +
                COALESCE(1.0 / (%(k)s + keyword_search.rank), 0.0) AS score,
                d.content
            FROM semantic_search
            FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
            JOIN documents d ON d.id = COALESCE(semantic_search.id, keyword_search.id)
            ORDER BY score DESC
            LIMIT 10
        """
        results = self.conn.execute(
            sql, {"query": query, "embedding": embedding, "k": k}
        ).fetchall()
        execution_time = time.time() - start_time
        print(f"Query execution time: {execution_time:.4f} seconds")
        return results
