-- ============================================================
-- 星驰科技 RAG · SQLite 结构化库 DDL（文档 §12.3）
-- 入库规则（§12.3）：
--   products 库存列不落库（降级）；
--   inventory 仅取 库存清单.tsv + xlsx/库存分布；
--   ticket 仅取 售后工单记录.csv；
--   customer 姓名脱敏、手机号哈希；
--   售后工单统计.xlsx 不落库。
-- ============================================================

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS product (
  model          TEXT PRIMARY KEY,
  name           TEXT NOT NULL,
  category       TEXT NOT NULL,
  unlock_methods TEXT,
  material       TEXT,
  sale_status    TEXT NOT NULL CHECK (sale_status IN ('已上市','预售')),
  warranty_years INTEGER
);

CREATE TABLE IF NOT EXISTS price (
  model          TEXT PRIMARY KEY REFERENCES product(model),
  factory_price  INTEGER,
  retail_price   INTEGER,
  gross_margin   TEXT,
  effective_date TEXT
);

CREATE TABLE IF NOT EXISTS inventory (
  sku            TEXT NOT NULL,
  warehouse      TEXT NOT NULL,
  qty            INTEGER NOT NULL,
  safety_qty     INTEGER NOT NULL,
  restock_status TEXT,
  updated_at     TEXT,
  PRIMARY KEY (sku, warehouse)
);

CREATE TABLE IF NOT EXISTS ticket (
  ticket_no     TEXT PRIMARY KEY,
  customer_id   TEXT NOT NULL,
  product_model TEXT,
  issue_type    TEXT,
  status        TEXT NOT NULL CHECK (status IN ('已解决','处理中','待客户反馈','已关闭')),
  created_date  TEXT,
  handle_hours  REAL
);

CREATE TABLE IF NOT EXISTS customer (
  customer_id   TEXT PRIMARY KEY,
  name_masked   TEXT,
  city          TEXT,
  phone_hash    TEXT,
  phone_tail    TEXT,
  register_date TEXT,
  tier          TEXT
);

CREATE TABLE IF NOT EXISTS faq_category (
  code                 TEXT PRIMARY KEY,
  category             TEXT,
  keywords             TEXT,
  dept                 TEXT,
  first_response_hours INTEGER
);

CREATE TABLE IF NOT EXISTS sales_monthly (
  month TEXT, model TEXT, qty INTEGER, amount INTEGER, mom TEXT,
  PRIMARY KEY (month, model)
);

CREATE TABLE IF NOT EXISTS sales_region (
  region TEXT PRIMARY KEY, share TEXT, top_model TEXT, yoy TEXT
);

CREATE TABLE IF NOT EXISTS sales_summary (
  model TEXT PRIMARY KEY, year_qty INTEGER, year_amount INTEGER, return_rate TEXT
);

CREATE INDEX IF NOT EXISTS idx_ticket_customer ON ticket(customer_id);
CREATE INDEX IF NOT EXISTS idx_inventory_sku ON inventory(sku);
