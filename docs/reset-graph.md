# 清空圖譜並重建 — 完整流程

## 步驟總覽

| 步驟 | 操作 | 說明 |
|------|------|------|
| 1 | 清空 Neo4j 圖資料 | 刪除所有節點與邊 |
| 2 | 清空本機 ckpt 快取 | 讓 Sync 層與 KAG SDK 層忘記已處理的檔案 |
| 3 | 重啟 app 容器 | 觸發重新 ingest |
| 4 | 確認重建進行中 | 觀察 log |

---

## 步驟 1 — 清空 Neo4j 圖資料

```bash
docker exec release-openspg-neo4j cypher-shell -u neo4j -p "neo4j@openspg" --database ucllab "MATCH (n) DETACH DELETE n"
```

確認已清空（應回傳 `0`）：

```bash
docker exec release-openspg-neo4j cypher-shell -u neo4j -p "neo4j@openspg" --database ucllab "MATCH (n) RETURN count(n) AS remaining"
```

---

## 步驟 2 — 清空本機 ckpt 快取

> **重要：** `ckpt/` 是 bind mount 到本機目錄，`docker compose rm` 不會清掉它。必須手動刪除。

在 PowerShell 執行：

```powershell
Remove-Item -Recurse -Force C:\Users\Han_Ucl\Desktop\UCL_KAG\ckpt\*
```

這會一次清除以下所有快取：

| 路徑 | 用途 |
|------|------|
| `ckpt/indexed.json` | Sync 層 MD5 紀錄（Drive 檔案去重） |
| `ckpt/KGWriter/` | KAG SDK — 已寫入 Neo4j 的節點/邊紀錄 |
| `ckpt/BatchVectorizer/` | KAG SDK — 已向量化的 chunk 紀錄 |
| `ckpt/LengthSplitter/` | KAG SDK — 已切段的文字紀錄 |
| `ckpt/SchemaConstraintExtractor/` | KAG SDK — 已做 NER 的段落紀錄 |
| `ckpt/TXTReader/` | KAG SDK — 已讀取的檔案紀錄 |

---

## 步驟 3 — 重啟 app 容器

```bash
docker compose restart app
```

---

## 步驟 4 — 確認重建進行中

```bash
docker compose logs -f app
```

正常應看到每個檔案逐一被處理：

```
[sync] indexing: 20260504研究生會議記錄.pdf
[sync] indexing: 20260427研究生會議記錄.pdf
...
[startup] initial ingest done: 15 documents
```

> 如果看到 `[sync] unchanged: ...` 表示步驟 2 沒有清乾淨，重新執行步驟 2。

---

## 常見錯誤

| 症狀 | 原因 | 解法 |
|------|------|------|
| 全部顯示 `unchanged`，ingest 0 docs | `ckpt/indexed.json` 未清除 | 重新執行步驟 2 |
| Log 有 ingest 但 Neo4j 查不到資料 | `ckpt/KGWriter/cache.db` 未清除，KGWriter 以為已寫入而跳過 | 重新執行步驟 2 |
| `docker compose rm app` 後快取仍存在 | ckpt 是 bind mount，刪容器不影響本機檔案 | 永遠用步驟 2 的 PowerShell 指令清除 |
