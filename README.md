# HMDA 2024 – Big Data Analysis Final Project (Group 5)

Ninh Ngọc Anh 11230734 · Bùi Thị Khánh Linh 11230756 · Nguyễn Thị Thu Phương 11230770

Dữ liệu chuẩn: **HMDA 2024 One-Year National Loan-Level Dataset**, freeze date 2026-06-02,
12,259,199 LAR records. URL và SHA-256 của LAR/TS được ghi trong `data/raw/source_manifest.json` sau khi chuẩn bị.
Hai engine độc lập (Pandas/pyarrow và PySpark 3.5.3) chạy cùng định nghĩa; counts so chính xác, keys so dạng chuỗi,
floats so theo tolerance. Correctness chỉ PASS khi đủ catalog đã ghim (164 raw aggregations + 67 final tables)
và cả 30 headline bắt buộc đều có giá trị và khớp.

## Câu hỏi nghiên cứu (theo proposal v1)
- **Main**: tỷ lệ từ chối (trong số hồ sơ đã có quyết định) và tỷ lệ originate thay đổi thế nào theo loan type, purpose, state, lender.
- **Q1** denial rate theo loan type × purpose, có nhất quán giữa các state không (Spearman, chuẩn hoá gián tiếp).
- **Q2** độ phân tán denial rate giữa các lender, và trong cùng một phân khúc sản phẩm.
- **Q3** thu nhập, loan amount, LTI, DTI, CLTV, giá trị tài sản liên quan thế nào đến denial (Cramér's V). Chỉ là association.
- **Q4** lãi suất / rate spread của khoản vay đã originate theo loan type/purpose/lien và đặc điểm khoản vay (OLS từ ma trận chuẩn). Chỉ là association.
- **Mở rộng dự báo (bước 11)**: denial trong nhóm đã có quyết định; chia train/val/test theo lender trước khi audit bằng outcome,
  LR so với gradient boosting và các ablation. Đây là xếp hạng hồi cứu: một số biến có thể phản ánh giai đoạn underwriting,
  còn lender profiles dùng cơ cấu cả năm và không được mô tả là sẵn có lúc nộp hồ sơ.

## Quy tắc dữ liệu (đã chốt)
- **Không đọc, không dùng** các biến nhân khẩu học (tuổi, giới, chủng tộc, dân tộc, của cả đồng vay) và các cột dẫn xuất của chúng,
  `tract_minority*`, `denial_reason`, `hoepa_status`, `purchaser_type`, `aus_`. Một "forbidden guard" (`FORBIDDEN_RE` trong `src/core/schema.py`) chặn các cột này ở cả hai engine.
- Không dùng dữ liệu ngoài để phân tích. File TS (Transmittal Sheet) chỉ dùng để đối chiếu số dòng và gán tên/cơ quan quản lý của lender;
  tài liệu ngoài chỉ được trích dẫn để kiểm chứng định nghĩa và đặt kết quả vào đúng bối cảnh.
- Quy tắc làm sạch giống hệt nhau ở hai engine: xem `cleaning_rules.md`.
- Dữ liệu giả lập (`scripts/make_synthetic.py`) chỉ để kiểm thử pipeline. Không có kết quả nào trong báo cáo được lấy từ đó.

## Cài đặt và dựng input (đường dẫn portable)
1. Python **3.12**, Java **17+** (Windows cần `HADOOP_HOME`/`winutils.exe` nếu môi trường Spark yêu cầu).
   Benchmark đi kèm được đo với OpenJDK **17.0.21**; đây là môi trường đo, không phải yêu cầu phải dùng đúng bản Java đó.
2. `python -m venv .venv`, activate môi trường, rồi `pip install -r requirements.txt`.
3. Tải đúng hai file 2024 One-Year từ URL chính thức dưới đây, sau đó chạy lệnh streaming; script không giải nén file text ra đĩa:
   ```
   python scripts/prepare_official_2024.py --lar-zip <2024_public_lar_one_year_pipe.zip> --ts-zip <2024_public_ts_one_year_pipe.zip>
   ```
   LAR: `https://files.ffiec.cfpb.gov/static-data/one-year/2024/2024_public_lar_one_year_pipe.zip`
   TS: `https://files.ffiec.cfpb.gov/static-data/one-year/2024/2024_public_ts_one_year_pipe.zip`
4. `config.yaml` đã dùng đường dẫn tương đối `data/raw/...`; chạy `python scripts/check_env.py`.

## Chạy
```
python run.py --list                      # liệt kê các bước
python run.py --steps all --engine both   # toàn bộ (bước 00-12)
python run.py --steps 05                  # chỉ chạy lại một bước
python run.py --steps 04-07 --engine pandas
python run.py --steps 06 --force          # bỏ qua cache
```
Mỗi bước có manifest gồm hash config và fingerprint toàn bộ code Python, nên thay code không tái sử dụng kết quả cũ.
Lần chạy nộp bài vẫn nên dùng `--force`; benchmark từ chối chạy nếu correctness hiện tại chưa PASS.

| Bước | Module | Nội dung |
|---|---|---|
| 00 | s00_inspect | schema, số file/dòng, đối chiếu TS |
| 01 | s01_prepare | dựng master Parquet phân tích (cờ và band, cờ trùng lặp, khoá lấy mẫu) |
| 02 | s02_quality | missing/status/ngoài ngưỡng, trùng lặp |
| 03 | s03_descriptive | mô tả tổng quan |
| 04–07 | s04_q1 … s07_q4 | Q1–Q4 (+ Main) |
| 08 | s08_robustness | biến thể loại trừ (lọc truy vấn, không xoá dữ liệu) |
| 09 | s09_correctness | so sánh mọi bảng giữa hai engine |
| 10 | s10_benchmark | hai workload (`lender_groupby`, `median_by_state`) trên cùng Pandas-master Parquet/layout/projection cho hai engine; compute-only và read+compute; physical plan Spark |
| 11 | s11_predictive | mở rộng dự báo |
| 12 | s12_report_assets | hình (cnsplots) và bảng ba đường kẻ (CSV + LaTeX), `figure_captions.md` |

Kết quả: `outputs/<engine>/raw`, `outputs/<engine>/tables`, `outputs/compare/tables`, `outputs/figures`, `outputs/tables_report`.
Dùng cả `raw/` và `tables/` để tái kiểm tra đầy đủ claim 231/231; chỉ `tables/` thì mới kiểm tra lại được 67 final pairs.
Dữ liệu trung gian: `data/interim`.

## Benchmark (bước 10)
- Hai workload: `lender_groupby` (đếm bản ghi, loan amount trung bình, số hồ sơ đã quyết định/từ chối và tỷ lệ từ chối theo lender) và `median_by_state` (trung vị chính xác của loan amount theo state, 55 nhóm gồm nhóm `NA`).
- Hai kịch bản: compute-only (Spark cache và count trước khi đo) và read+compute (đo cả việc đọc Parquet, xoá cache Spark trước mỗi lần chạy).
- Mẫu lồng nhau 25/50/100%, một lượt warm-up không tính giờ cho mỗi workload và kịch bản, 7 lần lặp có tính giờ theo thứ tự xoay có seed. Một ô được gắn nhãn "Tie / inconclusive" khi khoảng min–max của hai engine chồng nhau hoặc một engine không ổn định (max/min > 1.5). Cache của hệ điều hành không được xoá.
- Đầu ra: `outputs/compare/tables/benchmark_summary.csv`, `benchmark_timings_raw.csv`, `benchmark_memory.csv`, `benchmark_setup_cost.csv`, `benchmark_equivalence.csv`, `outputs/tables_report/table3_benchmark.csv`, `outputs/figures/fig5_engine_benchmark.pdf`.
- Cổng tương đương: mỗi ô của `median_by_state` được so Pandas với PySpark trước khi dùng thời gian; sáu bảng đầu ra được ghi vào `outputs/compare/benchmark_tables/` và thêm sáu dòng `kind=benchmark` vào `correctness_detail.csv`. Chạy lại bước 09 sẽ ghi đè các dòng này, khi đó phải chạy lại bước 10.

## Hình trong báo cáo
Các hình 0–6 do bước 12 tạo ra trong `outputs/figures/` (PDF và PNG). Hình `fig6_predictive` (mở rộng dự báo) nằm ở Phụ lục A của báo cáo và được đánh số Hình 7.

## Kiểm thử
`pytest tests -q` (23 test, gồm 4 test cho workload `median_by_state`) kiểm tra thống kê, guard cột cấm, `income=1111/1111.0`, DTI 50/60 ở cả hai engine,
duplicate semantics, correctness catalog/key/count/headline và end-to-end synthetic. Fixture tổng hợp có 1.200 dòng thuộc nhóm lặp và 600 bản sao dư.
Chạy thử correctness end-to-end với dữ liệu giả lập: `python scripts/make_synthetic.py` rồi
`python run.py --config config.test.yaml --steps 00-09 --engine both --force`. Benchmark, predictive và report assets dùng output full-data.

## Quy ước quan trọng
- `has_reported_exempt_value` là cờ record, còn `lenders_with_exempt_values` chỉ nghĩa là lender có ít nhất một record như vậy; không phải kết luận pháp lý rằng lender được miễn báo cáo.
- `is_dup` đánh dấu mọi dòng thuộc fingerprint lặp. Baseline giữ nguyên; sensitivity loại toàn bộ nhóm lặp và báo riêng số dòng trong nhóm với số bản sao dư.
- `~OUTSIDE_DOMAIN` nghĩa là không dùng được cho phép log/ratio/thống kê đang xét, không tự động là dữ liệu sai.

## Giới hạn cần nêu trong báo cáo
- Hồi quy Q4 báo cả sai số chuẩn cổ điển và sai số chuẩn gom theo lender (CR1); kết luận nên đọc từ cột gom theo lender. Chỉ diễn giải là association.
- Fingerprint lặp xác định trên các cột được phép đọc, nên chỉ là định nghĩa thực dụng; không tự động chứng minh hồ sơ trùng sai.
- Benchmark chỉ áp dụng cho hai workload và máy này; thời gian khởi tạo Spark tách riêng.
- Q3/Q4/dự báo không kết luận nhân quả, không đánh giá phân biệt đối xử (đã chủ động không dùng biến nhân khẩu học).
- Predictive section là retrospective risk ranking. Lender profiles dùng cơ cấu full-year trong từng split; ablation không dùng profile được báo riêng.
