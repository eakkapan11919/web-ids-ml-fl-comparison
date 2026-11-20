Streamlit Camera-ready V3: โฟลเดอร์เตรียมอัปโหลด GitHub
1) app.py : หน้าเว็บเดิมที่ปรับใช้โมเดลใหม่ พร้อม Research Evidence
2) ids_pipeline.py : อ่านไฟล์และทำนายเป็นชุด 5,000 แถว
3) xgboost_models.joblib, lightgbm_models.joblib, validation_preprocessing.joblib : โมเดลจริงจากการทดลอง seed42
4) model_manifest.json : รายละเอียด Dataset Split/Threshold
5) research_evidence/ : ตารางและกราฟจากผลทดลองที่บันทึกแล้ว
6) .streamlit/config.toml : ตั้งขนาดไฟล์อัปโหลดสูงสุด 250 MB
7) demo_upload_sample.parquet : ตัวอย่างสำหรับสาธิต
8) README.md / requirements.txt : คู่มือและไลบรารี
หลักฐานการตรวจ: อ่านและทำนาย Botnet ต้นฉบับ 771,587 แถวและสร้าง ZIP ผลครบ
ยังไม่ได้พิสูจน์ว่า Cloud รองรับทุกไฟล์/ทุกขนาด ต้อง Deploy Branch ทดสอบแยกจาก main ก่อน
ห้ามนำ Dataset ต้นฉบับและไฟล์ 03_RESULTS จำนวนมากขึ้น GitHub โดยไม่จำเป็น
