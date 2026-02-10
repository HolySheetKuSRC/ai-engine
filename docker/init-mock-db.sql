-- Initialize mock database schema and seed data
-- สร้างตารางจำลอง (ตาม Schema ของคุณ)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR NOT NULL,
    email VARCHAR NOT NULL UNIQUE,
    role VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS universities (
    id BIGSERIAL PRIMARY KEY,
    name_th VARCHAR,
    name_en VARCHAR
);

CREATE TABLE IF NOT EXISTS sheets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    seller_id UUID NOT NULL,
    title VARCHAR NOT NULL,
    description TEXT,
    file_url VARCHAR NOT NULL, -- ใน Dev อาจจะใส่เป็น Path file ในเครื่อง
    price NUMERIC DEFAULT 0,
    summary TEXT,
    ai_assessment TEXT,
    tags TEXT,
    ocr_content TEXT,
    page_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ใส่ข้อมูล Mock (Seeding)
INSERT INTO users (name, email, role) VALUES 
('Test Seller', 'seller@example.com', 'SELLER'),
('Test Buyer', 'buyer@example.com', 'USER');

INSERT INTO universities (name_th, name_en) VALUES 
('จุฬาลงกรณ์มหาวิทยาลัย', 'Chulalongkorn University'),
('มหาวิทยาลัยเกษตรศาสตร์', 'Kasetsart University');

-- ข้อมูลนี้สำคัญมากเพื่อให้ AI Code ของคุณมีของไป test
INSERT INTO sheets (seller_id, title, description, file_url, price) 
VALUES 
((SELECT id FROM users LIMIT 1), 'สรุป Python 101', 'ชีทสรุปพื้นฐานภาษา Python สำหรับ Data Science', 'https://mock-url.com/sheet1.pdf', 50.00);