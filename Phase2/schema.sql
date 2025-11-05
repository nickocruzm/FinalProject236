-- Database Schema for Hotel Booking Project
-- This schema supports three tables: hotel_data, customer_data, and unified_data

-- Drop tables if they exist
DROP TABLE IF EXISTS hotel_data CASCADE;
DROP TABLE IF EXISTS customer_data CASCADE;
DROP TABLE IF EXISTS unified_data CASCADE;

-- Table 1: Hotel Booking Dataset (2015-2016)
CREATE TABLE hotel_data (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255),
    country VARCHAR(100),
    hotel_type VARCHAR(100),
    arrival_year INTEGER,
    arrival_month INTEGER,
    arrival_week_number INTEGER,
    arrival_day INTEGER,
    arrival_date DATE,
    lead_time INTEGER,
    no_of_weekend_nights INTEGER,
    no_of_week_nights INTEGER,
    market_segment_type VARCHAR(100),
    avg_price_per_room DECIMAL(10, 2),
    is_canceled INTEGER
);

-- Table 2: Customer Reservations Dataset (2017-2018)
CREATE TABLE customer_data (
    id SERIAL PRIMARY KEY,
    booking_id VARCHAR(100) UNIQUE,
    arrival_year INTEGER,
    arrival_month INTEGER,
    arrival_day INTEGER,
    arrival_date DATE,
    lead_time INTEGER,
    no_of_weekend_nights INTEGER,
    no_of_week_nights INTEGER,
    market_segment_type VARCHAR(100),
    avg_price_per_room DECIMAL(10, 2),
    is_canceled INTEGER
);

-- Table 3: Unified Dataset (Combined 2015-2018)
CREATE TABLE unified_data (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255),
    country VARCHAR(100),
    hotel_type VARCHAR(100),
    arrival_week_number INTEGER,
    booking_id VARCHAR(100),
    arrival_year INTEGER,
    arrival_month INTEGER,
    arrival_day INTEGER,
    arrival_date DATE,
    lead_time INTEGER,
    no_of_weekend_nights INTEGER,
    no_of_week_nights INTEGER,
    market_segment_type VARCHAR(100),
    avg_price_per_room DECIMAL(10, 2),
    is_canceled INTEGER
);

-- Create indexes for better query performance
CREATE INDEX idx_hotel_arrival_date ON hotel_data(arrival_date);

CREATE INDEX idx_customer_arrival_date ON customer_data(arrival_date);
CREATE INDEX idx_customer_booking_id ON customer_data(booking_id);

CREATE INDEX idx_unified_arrival_date ON unified_data(arrival_date);
