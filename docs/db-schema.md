 # Growthsheet Backend – Database Structure (PostgreSQL)

> Source of truth: JPA entities in each service. Hibernate `ddl-auto=update` builds/updates tables at runtime.

## Auth Service (auth-service)

### Table: `users`
- `id` UUID (PK)
- `name` varchar (NOT NULL)
- `email` varchar (NOT NULL, UNIQUE)
- `password` varchar (NOT NULL)
- `enabled` boolean (NOT NULL, default true)
- `user_photo_url` varchar (nullable)
- `role` varchar (NOT NULL) — enum `UserRole`

### Table: `otp_token`
- `id` UUID (PK)
- `email` varchar (NOT NULL)
- `otp` varchar (NOT NULL)
- `expires_at` timestamp (NOT NULL)

## Order Service (order-service)

### Table: `carts`
- `id` UUID (PK)
- `user_id` UUID
- `total_price` numeric

### Table: `cart_items`
- `id` UUID (PK)
- `sheet_id` UUID
- `sheet_name` varchar
- `seller_name` varchar
- `price` numeric
- `cart_id` UUID (FK → `carts.id`)

**Relationship**
- `carts` (1) — (N) `cart_items`

### Table: `orders`
- `id` UUID (PK)
- `user_id` UUID
- `status` varchar
- `total_price` numeric

### Table: `order_items`
- `id` UUID (PK)
- `sheet_id` UUID
- `sheet_name` varchar
- `seller_name` varchar
- `price` numeric
- `order_id` UUID (FK → `orders.id`)

**Relationship**
- `orders` (1) — (N) `order_items`

## Product Service (product-service)

### Table: `categories`
- `id` bigint (PK, identity)
- `name` varchar (NOT NULL, UNIQUE)

### Table: `hashtags`
- `id` bigint (PK, identity)
- `name` varchar(50) (NOT NULL, UNIQUE)

### Table: `universities`
- `id` bigint (PK, identity)
- `name_th` varchar
- `name_en` varchar

### Table: `sheets`
- `id` UUID (PK)
- `seller_id` UUID (NOT NULL)
- `university_id` bigint (FK → `universities.id`)
- `category_id` bigint (FK → `categories.id`)
- `title` varchar (NOT NULL)
- `course_code` varchar (NOT NULL)
- `course_name` varchar (NOT NULL)
- `faculty` varchar
- `study_year` int (NOT NULL)
- `academic_year` varchar (NOT NULL)
- `description` text
- `price` numeric (NOT NULL, default 0)
- `file_url` varchar (NOT NULL)
- `page_count` int
- `status` varchar (enum `SheetStatus`, default `PENDING`)
- `admin_note` varchar
- `is_published` boolean (NOT NULL, default true)
- `average_rating` numeric (default 0)
- `review_count` int (default 0)
- `created_at` timestamp
- `updated_at` timestamp

### Table: `sheet_images`
- `id` UUID (PK)
- `image_url` varchar (NOT NULL)
- `sort_order` int
- `sheet_id` UUID (FK → `sheets.id`, NOT NULL)

### Table: `sheet_hashtags` (join table)
- `sheet_id` UUID (FK → `sheets.id`)
- `hashtag_id` bigint (FK → `hashtags.id`)

### Table: `sheet_reviews`
- `id` UUID (PK)
- `sheet_id` UUID
- `user_id` UUID
- `rating` int
- `comment` varchar
- `created_at` timestamp
- `updated_at` timestamp

### Table: `users` (read model in product-service)
- `id` UUID (PK)
- `name` varchar
- `username` varchar
- `email` varchar

**Relationships**
- `universities` (1) — (N) `sheets`
- `categories` (1) — (N) `sheets`
- `sheets` (1) — (N) `sheet_images`
- `sheets` (N) — (N) `hashtags` via `sheet_hashtags`

## Redis (runtime cache/session)
- Used by `auth-service` and `apigateway-service` for session/token storage (not relational).

## Notes
- Column names follow JPA defaults unless specified by `@Column(name=...)`.
- Actual DDL may differ slightly depending on Hibernate naming strategy.
