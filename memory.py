"""
记忆管理模块
实现Agent的短期和长期记忆功能
"""
import sqlite3
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

@dataclass
class Memory:
    """记忆数据结构"""
    id: Optional[int] = None
    content: str = ""
    memory_type: str = "conversation"  # conversation, experience, knowledge
    importance: float = 0.5  # 重要性评分 0-1
    timestamp: float = 0.0
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

class MemoryManager:
    """记忆管理器"""
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建记忆表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                memory_type TEXT DEFAULT 'conversation',
                importance REAL DEFAULT 0.5,
                timestamp REAL NOT NULL,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)")
        
        conn.commit()
        conn.close()
    
    def add_memory(self, memory: Memory) -> int:
        """添加记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO memories (content, memory_type, importance, timestamp, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            memory.content,
            memory.memory_type,
            memory.importance,
            memory.timestamp,
            json.dumps(memory.tags),
            json.dumps(memory.metadata)
        ))
        
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return memory_id or 0
    
    def get_recent_memories(self, limit: int = 10, memory_type: Optional[str] = None) -> List[Memory]:
        """获取最近的记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if memory_type:
            cursor.execute("""
                SELECT * FROM memories 
                WHERE memory_type = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (memory_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM memories 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]
    
    def get_important_memories(self, min_importance: float = 0.7, limit: int = 20) -> List[Memory]:
        """获取重要记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM memories 
            WHERE importance >= ?
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        """, (min_importance, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]
    
    def search_memories(self, query: str, limit: int = 10) -> List[Memory]:
        """搜索记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM memories 
            WHERE content LIKE ?
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        """, (f"%{query}%", limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]
    
    def update_memory_importance(self, memory_id: int, new_importance: float):
        """更新记忆重要性"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE memories 
            SET importance = ?
            WHERE id = ?
        """, (new_importance, memory_id))
        
        conn.commit()
        conn.close()
    
    def cleanup_old_memories(self, days: int = 30, keep_important: bool = True):
        """清理旧记忆"""
        cutoff_timestamp = time.time() - (days * 24 * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if keep_important:
            cursor.execute("""
                DELETE FROM memories 
                WHERE timestamp < ? AND importance < 0.7
            """, (cutoff_timestamp,))
        else:
            cursor.execute("""
                DELETE FROM memories 
                WHERE timestamp < ?
            """, (cutoff_timestamp,))
        
        conn.commit()
        conn.close()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总记忆数
        cursor.execute("SELECT COUNT(*) FROM memories")
        total_count = cursor.fetchone()[0]
        
        # 按类型统计
        cursor.execute("""
            SELECT memory_type, COUNT(*) 
            FROM memories 
            GROUP BY memory_type
        """)
        type_stats = dict(cursor.fetchall())
        
        # 重要记忆数
        cursor.execute("SELECT COUNT(*) FROM memories WHERE importance >= 0.7")
        important_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_memories": total_count,
            "important_memories": important_count,
            "type_distribution": type_stats,
            "database_path": self.db_path
        }
    
    def _row_to_memory(self, row) -> Memory:
        """将数据库行转换为Memory对象"""
        return Memory(
            id=row[0],
            content=row[1],
            memory_type=row[2],
            importance=row[3],
            timestamp=row[4],
            tags=json.loads(row[5]),
            metadata=json.loads(row[6])
        )
    
    def export_memories(self, filepath: str, memory_type: Optional[str] = None):
        """导出记忆到文件"""
        if memory_type:
            memories = self.get_recent_memories(limit=1000, memory_type=memory_type)
        else:
            memories = self.get_recent_memories(limit=1000)
        
        export_data = [asdict(memory) for memory in memories]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def import_memories(self, filepath: str):
        """从文件导入记忆"""
        with open(filepath, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        for memory_data in import_data:
            memory_data.pop('id', None)  # 移除ID，让数据库自动生成
            memory = Memory(**memory_data)
            self.add_memory(memory)