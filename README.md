# UCL Lab KAG Assistant

UCL 實驗室知識增強生成（KAG）問答系統，結合 Neo4j 知識圖譜推理與向量語意搜尋，回答實驗室資源相關問題。文件從 Google Drive 自動匯入，透過 OpenSPG-KAG SDK 建立索引。

## 系統架構

| 服務 | 說明 |
|------|------|
| `mysql` | OpenSPG 控制平面的 Metadata 儲存 |
| `neo4j` | 知識圖譜 + 向量索引 |
| `minio` | KAG 物件儲存 |
| `openspg` | Java-based OpenSPG 控制平面 |
| `app` | FastAPI 應用（port 8001）|

## 快速開始

### 1. 環境設定

```bash
cp .env.example .env
# 填入以下必要欄位：
# OPENAI_BASE_URL, OPENAI_API_KEY
# GDRIVE_FOLDER_ID
# GOOGLE_SERVICE_ACCOUNT_JSON
```

### 2. 啟動服務

```bash
docker compose up --build
```

### 3. 確認服務狀態

```bash
curl http://localhost:8001/api/health
curl http://localhost:8001/api/status
```

### 4. 發送問題

```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "GPU 叢集由誰管理？"}'
```

## 知識圖譜 Schema

共 12 種實體類型，定義於 `schema/UCLLab.schema`：

`Person`、`Project`、`Task`、`MeetingRecord`、`Decision`、`ResearchPlan`、`Patent`、`ComputeResource`、`Equipment`、`Course`、`ExternalParty`、`Competition`

## 專案結構

```
backend/
├── app/
│   ├── main.py              # FastAPI 進入點
│   ├── config.py            # 環境設定
│   ├── schemas.py           # API 資料模型
│   ├── prompts/
│   │   ├── ucl_lab_ner.py       # NER Prompt（實體抽取）
│   │   └── ucl_lab_relation.py  # 關係抽取 Prompt
│   └── services/
│       ├── kag_service.py       # KAG Builder + Solver
│       ├── gdrive_loader.py     # Google Drive 存取
│       ├── sync_service.py      # 定期同步排程
│       └── document_parser.py   # PDF/DOCX/XLSX/PPTX 解析
schema/
└── UCLLab.schema            # OpenSPG DSL Schema 定義
frontend/                    # 靜態前端頁面
```

## 注意事項

- 完整重置（`docker compose down -v`）後，須刪除 `/app/ckpt/` 否則 KGWriter 會跳過重新匯入
- Neo4j 資料庫名稱為 `ucllab`（非預設的 `neo4j`）
- `kag_config.yaml` 為 template，由 `entrypoint.sh` 在容器啟動時透過 `envsubst` 生成，請勿直接編輯容器內的版本
