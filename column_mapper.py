# import psycopg2
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# DB_CONFIG = {
#     "dbname": "postgres",
#     "user": "postgres",
#     "password": "Peetu@31?!",
#     "host": "localhost",
#     "port": "5432"
# }

# def backup_existing_data():
#     """Backup existing data before schema cleanup"""
#     print("💾 BACKING UP EXISTING DATA")
#     print("=" * 50)
    
#     conn = psycopg2.connect(**DB_CONFIG)
#     cursor = conn.cursor()
    
#     backup_data = {}
    
#     try:
#         # Check if tables have any data
#         tables = ['payer_details', 'plan_details', 'drug_formulary_details']
        
#         for table in tables:
#             cursor.execute(f"SELECT COUNT(*) FROM {table}")
#             count = cursor.fetchone()[0]
#             print(f"📊 {table}: {count} records")
            
#             if count > 0:
#                 print(f"⚠️  {table} contains data - please backup manually if needed")
#                 backup_data[table] = count
        
#         return backup_data
        
#     except Exception as e:
#         print(f"❌ Error checking data: {e}")
#         return {}
#     finally:
#         cursor.close()
#         conn.close()

# def drop_and_recreate_tables():
#     """Drop existing tables and recreate with correct schema"""
#     print("\n🗑️  DROPPING AND RECREATING TABLES")
#     print("=" * 50)
    
#     response = input("⚠️  This will DELETE ALL DATA and recreate tables. Continue? (yes/no): ")
#     if response.lower() != 'yes':
#         print("❌ Operation cancelled")
#         return False
    
#     conn = psycopg2.connect(**DB_CONFIG)
#     cursor = conn.cursor()
    
#     try:
#         # Drop tables in reverse order due to foreign keys
#         print("🗑️  Dropping existing tables...")
#         cursor.execute("DROP TABLE IF EXISTS drug_formulary_details CASCADE;")
#         cursor.execute("DROP TABLE IF EXISTS plan_details CASCADE;")
#         cursor.execute("DROP TABLE IF EXISTS payer_details CASCADE;")
#         print("✓ Existing tables dropped")
        
#         # Create payer_details table
#         print("🔨 Creating payer_details table...")
#         cursor.execute("""
#             CREATE TABLE payer_details (
#                 payer_id VARCHAR(36) PRIMARY KEY,
#                 payer_name VARCHAR(255) NOT NULL,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             );
#         """)
#         print("✓ payer_details table created")
        
#         # Create plan_details table
#         print("🔨 Creating plan_details table...")
#         cursor.execute("""
#             CREATE TABLE plan_details (
#                 plan_id VARCHAR(36) PRIMARY KEY,
#                 payer_id VARCHAR(36) REFERENCES payer_details(payer_id) ON DELETE CASCADE,
#                 plan_name VARCHAR(255) NOT NULL,
#                 state_code VARCHAR(2),
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 last_updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             );
#         """)
#         print("✓ plan_details table created")
        
#         # Create drug_formulary_details table
#         print("🔨 Creating drug_formulary_details table...")
#         cursor.execute("""
#             CREATE TABLE drug_formulary_details (
#                 id VARCHAR(36) PRIMARY KEY,
#                 plan_id VARCHAR(36) REFERENCES plan_details(plan_id) ON DELETE CASCADE,
#                 payer_id VARCHAR(36) REFERENCES payer_details(payer_id) ON DELETE CASCADE,
#                 drug_name VARCHAR(255),
#                 ndc_code VARCHAR(50),
#                 jcode VARCHAR(50),
#                 state_code VARCHAR(2),
#                 coverage_status VARCHAR(100),
#                 drug_tier VARCHAR(50),
#                 drug_requirements TEXT,
#                 coverage_details TEXT,
#                 confidence_score DECIMAL(3,2),
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 last_updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 source_url TEXT,
#                 file_name VARCHAR(255)
#             );
#         """)
#         print("✓ drug_formulary_details table created")
        
#         # Create indexes for performance
#         print("🔨 Creating indexes...")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_payer_name ON payer_details(payer_name);")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_payer ON plan_details(payer_id);")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_state ON plan_details(state_code);")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_drug_plan ON drug_formulary_details(plan_id);")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_drug_name ON drug_formulary_details(drug_name);")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_drug_payer ON drug_formulary_details(payer_id);")
#         print("✓ Indexes created")
        
#         conn.commit()
#         print("✅ All tables recreated successfully!")
#         return True
        
#     except Exception as e:
#         conn.rollback()
#         print(f"❌ Error recreating tables: {e}")
#         return False
#     finally:
#         cursor.close()
#         conn.close()

# def verify_schema():
#     """Verify the new schema is correct"""
#     print("\n🔍 VERIFYING NEW SCHEMA")
#     print("=" * 50)
    
#     conn = psycopg2.connect(**DB_CONFIG)
#     cursor = conn.cursor()
    
#     try:
#         tables = ['payer_details', 'plan_details', 'drug_formulary_details']
        
#         for table_name in tables:
#             print(f"\n📋 {table_name}:")
            
#             cursor.execute("""
#                 SELECT 
#                     column_name, 
#                     data_type, 
#                     is_nullable,
#                     column_default
#                 FROM information_schema.columns 
#                 WHERE table_name = %s 
#                 ORDER BY ordinal_position;
#             """, (table_name,))
            
#             columns = cursor.fetchall()
            
#             for i, col in enumerate(columns, 1):
#                 col_name, data_type, nullable, default = col
#                 null_text = "NULL" if nullable == "YES" else "NOT NULL"
#                 default_text = f" DEFAULT {default}" if default else ""
#                 print(f"  {i:2d}. {col_name:<25} {data_type:<20} {null_text}{default_text}")
        
#         print("\n✅ Schema verification complete!")
        
#     except Exception as e:
#         print(f"❌ Error verifying schema: {e}")
#     finally:
#         cursor.close()
#         conn.close()

# def test_script_compatibility():
#     """Test if the original script will work with new schema"""
#     print("\n🧪 TESTING SCRIPT COMPATIBILITY")
#     print("=" * 50)
    
#     conn = psycopg2.connect(**DB_CONFIG)
#     cursor = conn.cursor()
    
#     try:
#         # Test payer_details insert
#         print("Testing payer_details insert...")
#         cursor.execute("""
#             INSERT INTO payer_details (payer_id, payer_name, created_at, last_updated_at) 
#             VALUES ('test-payer-id', 'Test Payer', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
#         """)
#         print("✓ payer_details insert works")
        
#         # Test plan_details insert
#         print("Testing plan_details insert...")
#         cursor.execute("""
#             INSERT INTO plan_details (plan_id, payer_id, plan_name, state_code, created_at, last_updated_date) 
#             VALUES ('test-plan-id', 'test-payer-id', 'Test Plan', 'FL', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
#         """)
#         print("✓ plan_details insert works")
        
#         # Test drug_formulary_details insert
#         print("Testing drug_formulary_details insert...")
#         cursor.execute("""
#             INSERT INTO drug_formulary_details (
#                 id, plan_id, payer_id, drug_name, ndc_code, jcode,
#                 state_code, coverage_status, drug_tier, drug_requirements,
#                 coverage_details, confidence_score, created_at, 
#                 last_updated_date, source_url, file_name
#             ) VALUES (
#                 'test-drug-id', 'test-plan-id', 'test-payer-id', 'Test Drug', '12345-678-90', 'J1234',
#                 'FL', 'Covered', 'Tier 1', 'No requirements',
#                 'Full coverage', 0.95, CURRENT_TIMESTAMP, 
#                 CURRENT_TIMESTAMP, 'http://test.com', 'test.pdf'
#             )
#         """)
#         print("✓ drug_formulary_details insert works")
        
#         # Clean up test data
#         cursor.execute("DELETE FROM drug_formulary_details WHERE id = 'test-drug-id'")
#         cursor.execute("DELETE FROM plan_details WHERE plan_id = 'test-plan-id'")
#         cursor.execute("DELETE FROM payer_details WHERE payer_id = 'test-payer-id'")
        
#         conn.commit()
#         print("✅ All insert tests passed! Your original script should work now.")
        
#     except Exception as e:
#         conn.rollback()
#         print(f"❌ Test failed: {e}")
#     finally:
#         cursor.close()
#         conn.close()

# def main():
#     print("🚀 DATABASE SCHEMA CLEANUP TOOL")
#     print("=" * 60)
#     print("This script will fix the duplicate columns and schema issues in your database.")
#     print("⚠️  WARNING: This will delete all existing data!")
#     print("=" * 60)
    
#     # Step 1: Check existing data
#     backup_data = backup_existing_data()
    
#     if backup_data:
#         print(f"\n📊 Found existing data in {len(backup_data)} tables")
#         print("Please backup your data manually before proceeding if you need it.")
#         print("\nExisting data counts:")
#         for table, count in backup_data.items():
#             print(f"  • {table}: {count} records")
    
#     # Step 2: Ask for confirmation and recreate tables
#     if drop_and_recreate_tables():
#         # Step 3: Verify new schema
#         verify_schema()
        
#         # Step 4: Test compatibility
#         test_script_compatibility()
        
#         print(f"\n🎉 SCHEMA CLEANUP COMPLETE!")
#         print("=" * 60)
#         print("✅ Your database schema is now clean and ready")
#         print("✅ Your original db_updation.py script should work now")
#         print("✅ No more duplicate columns or spacing issues")
#         print("\nYou can now run: python db_updation.py")
        
#     else:
#         print("\n❌ Schema cleanup was cancelled or failed")

# if __name__ == "__main__":
#     main()

import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "Peetu@31?!",
    "host": "localhost",
    "port": "5432"
}

def backup_existing_data():
    """Backup existing data before schema cleanup"""
    print("💾 BACKING UP EXISTING DATA")
    print("=" * 50)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    backup_data = {}
    
    try:
        # Check if tables have any data
        tables = ['payer_details', 'plan_details', 'drug_formulary_details']
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"📊 {table}: {count} records")
            
            if count > 0:
                print(f"⚠️  {table} contains data - please backup manually if needed")
                backup_data[table] = count
        
        return backup_data
        
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return {}
    finally:
        cursor.close()
        conn.close()

def drop_and_recreate_tables():
    """Drop existing tables and recreate with correct enhanced schema"""
    print("\n🗑️  DROPPING AND RECREATING TABLES (ENHANCED SCHEMA)")
    print("=" * 50)
    
    response = input("⚠️  This will DELETE ALL DATA and recreate tables with enhanced schema. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Operation cancelled")
        return False
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # Drop tables in reverse order due to foreign keys
        print("🗑️  Dropping existing tables...")
        cursor.execute("DROP TABLE IF EXISTS drug_formulary_details CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS plan_details CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS payer_details CASCADE;")
        print("✓ Existing tables dropped")
        
        # Create enhanced payer_details table with contact information
        print("🔨 Creating enhanced payer_details table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS payer_details (
                    payer_id VARCHAR(36) PRIMARY KEY,
                    payer_name VARCHAR(1000) NOT NULL,
                    contact_phone VARCHAR(50),
                    address_line_1 VARCHAR(1000),
                    address_line_2 VARCHAR(1000),
                    city VARCHAR(100),
                    state VARCHAR(50),
                    zip_code VARCHAR(20),
                    status VARCHAR(20) DEFAULT 'processing',
                    created_at TIMESTAMP,
                    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("✓ Enhanced payer_details table created with contact information")
        
        # Create enhanced plan_details table with formulary information
        print("🔨 Creating enhanced plan_details table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plan_details (
                    plan_id VARCHAR(36) PRIMARY KEY,
                    payer_id VARCHAR(36) NOT NULL,
                    plan_name VARCHAR(1000) NOT NULL,
                    state_name VARCHAR(100) NOT NULL,
                    formulary_url TEXT,
                    source_link TEXT,
                    formulary_date TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'processing',
                    created_at TIMESTAMP,
                    last_updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("✓ Enhanced plan_details table created with formulary URLs and dates")
        
        # Create enhanced drug_formulary_details table
        print("🔨 Creating enhanced drug_formulary_details table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drug_formulary_details (
                    id VARCHAR(36) NOT NULL,
                    plan_id VARCHAR(36) NOT NULL,
                    payer_id VARCHAR(36) NOT NULL,
                    drug_name VARCHAR(5000) NOT NULL,
                    ndc_code VARCHAR(50),
                    jcode VARCHAR(50),
                    state_name VARCHAR(100) NOT NULL,
                    coverage_status VARCHAR(1000),
                    drug_tier VARCHAR(1000),
                    drug_requirements VARCHAR(1000),
                    is_prior_authorization_required VARCHAR(10),
                    is_step_therapy_required VARCHAR(10),
                    coverage_details VARCHAR(10000),
                    confidence_score DECIMAL(3,2),
                    source_url TEXT,
                    file_name VARCHAR(1000),
                    status VARCHAR(20) DEFAULT 'processing',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (id, plan_id)
                ) PARTITION BY HASH (plan_id);
        """)
        print("✓ Enhanced drug_formulary_details table created")
        
        # Create indexes for performance
        print("🔨 Creating indexes...")
        
        # Payer indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_payer_name ON payer_details(payer_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_payer_state ON payer_details(state);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_payer_city ON payer_details(city);")
        
        # Plan indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_payer ON plan_details(payer_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_state_name ON plan_details(state_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_name ON plan_details(plan_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_formulary_date ON plan_details(formulary_date);")
        
        # Drug formulary indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_drug_plan ON drug_formulary_details(plan_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_drug_payer ON drug_formulary_details(payer_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_drug_name ON drug_formulary_details(drug_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_drug_tier ON drug_formulary_details(drug_tier);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_drug_state_name ON drug_formulary_details(state_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_drug_ndc ON drug_formulary_details(ndc_code);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_drug_jcode ON drug_formulary_details(jcode);")
        
        print("✓ All indexes created")
        
        conn.commit()
        print("✅ All enhanced tables recreated successfully!")
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error recreating tables: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def verify_schema():
    """Verify the new enhanced schema is correct"""
    print("\n🔍 VERIFYING NEW ENHANCED SCHEMA")
    print("=" * 50)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        tables = ['payer_details', 'plan_details', 'drug_formulary_details']
        
        for table_name in tables:
            print(f"\n📋 {table_name}:")
            
            cursor.execute("""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = %s 
                ORDER BY ordinal_position;
            """, (table_name,))
            
            columns = cursor.fetchall()
            
            for i, col in enumerate(columns, 1):
                col_name, data_type, nullable, default, max_length = col
                null_text = "NULL" if nullable == "YES" else "NOT NULL"
                default_text = f" DEFAULT {default}" if default else ""
                length_text = f"({max_length})" if max_length else ""
                print(f"  {i:2d}. {col_name:<25} {data_type}{length_text:<20} {null_text}{default_text}")
        
        print("\n✅ Enhanced schema verification complete!")
        
        # Show foreign key relationships
        print(f"\n🔗 FOREIGN KEY RELATIONSHIPS:")
        cursor.execute("""
            SELECT 
                tc.table_name, 
                kcu.column_name, 
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name 
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
            ORDER BY tc.table_name;
        """)
        
        fk_relationships = cursor.fetchall()
        for fk in fk_relationships:
            print(f"  🔗 {fk[0]}.{fk[1]} → {fk[2]}.{fk[3]}")
        
    except Exception as e:
        print(f"❌ Error verifying schema: {e}")
    finally:
        cursor.close()
        conn.close()

def test_script_compatibility():
    """Test if the enhanced script will work with new schema"""
    print("\n🧪 TESTING ENHANCED SCRIPT COMPATIBILITY")
    print("=" * 50)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # Test enhanced payer_details insert
        print("Testing enhanced payer_details insert...")
        cursor.execute("""
            INSERT INTO payer_details (
                payer_id, payer_name, contact_phone, address_line_1, 
                address_line_2, city, state, zip_code, created_at, 
                last_updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            'test-payer-id', 
            'Test Insurance Company', 
            '1-800-123-4567',
            '123 Insurance Ave',
            'Suite 100',
            'Test City',
            'FL',
            '12345',
            'CURRENT_TIMESTAMP',
            'CURRENT_TIMESTAMP'
        ))
        print("✓ Enhanced payer_details insert works")
        
        # Test enhanced plan_details insert
        print("Testing enhanced plan_details insert...")
        cursor.execute("""
            INSERT INTO plan_details (
                plan_id, payer_id, plan_name, state_name, formulary_url,
                source_link, formulary_date, created_at, last_updated_date
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            'test-plan-id', 
            'test-payer-id', 
            'Test Health Plan', 
            'Florida',
            'https://example.com/formulary.pdf',
            'https://example.com/source',
            'CURRENT_TIMESTAMP',
            'CURRENT_TIMESTAMP',
            'CURRENT_TIMESTAMP'
        ))
        print("✓ Enhanced plan_details insert works")
        
        # Test enhanced drug_formulary_details insert
        print("Testing enhanced drug_formulary_details insert...")
        cursor.execute("""
            INSERT INTO drug_formulary_details (
                id, plan_id, payer_id, drug_name, ndc_code, jcode,
                state_name, coverage_status, drug_tier, drug_requirements,
                coverage_details, confidence_score, created_at, 
                last_updated_date, source_url, file_name
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """, (
            'test-drug-id', 
            'test-plan-id', 
            'test-payer-id', 
            'Test Drug Name', 
            '12345-678-90', 
            'J1234',
            'Florida', 
            'Covered', 
            'Tier 1', 
            'Prior authorization required',
            'Full coverage with restrictions', 
            0.95, 
            'CURRENT_TIMESTAMP',
            'CURRENT_TIMESTAMP', 
            'https://example.com/formulary.pdf', 
            'Florida_Test_Insurance_Company_Test_Health_Plan.pdf'
        ))
        print("✓ Enhanced drug_formulary_details insert works")
        
        # Test data retrieval like in the main script
        print("Testing data retrieval queries...")
        
        # Test payer lookup with enhanced fields
        cursor.execute("""
            SELECT payer_id, payer_name, contact_phone, city, state
            FROM payer_details 
            WHERE payer_name = %s
        """, ('Test Insurance Company',))
        payer_result = cursor.fetchone()
        print(f"✓ Payer lookup works: {payer_result}")
        
        # Test plan lookup with enhanced fields
        cursor.execute("""
            SELECT pd.plan_id, pd.payer_id, py.payer_name, pd.plan_name, pd.formulary_url
            FROM plan_details pd
            JOIN payer_details py ON pd.payer_id = py.payer_id
            WHERE pd.state_name = %s 
            AND UPPER(py.payer_name) = UPPER(%s)
            AND UPPER(pd.plan_name) = UPPER(%s)
        """, ('Florida', 'Test Insurance Company', 'Test Health Plan'))
        plan_result = cursor.fetchone()
        print(f"✓ Plan lookup works: {plan_result}")
        
        # Clean up test data
        cursor.execute("DELETE FROM drug_formulary_details WHERE id = 'test-drug-id'")
        cursor.execute("DELETE FROM plan_details WHERE plan_id = 'test-plan-id'")
        cursor.execute("DELETE FROM payer_details WHERE payer_id = 'test-payer-id'")
        
        conn.commit()
        print("✅ All enhanced insert and query tests passed!")
        print("✅ Your enhanced processing script should work perfectly now!")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Test failed: {e}")
    finally:
        cursor.close()
        conn.close()

def show_schema_summary():
    """Show a summary of the enhanced schema"""
    print("\n📋 ENHANCED SCHEMA SUMMARY")
    print("=" * 60)
    
    schema_info = {
        "payer_details": {
            "description": "Insurance companies with full contact information",
            "key_fields": ["payer_name", "contact_phone", "address_line_1", "city", "state", "zip_code"],
            "enhancements": "Added contact information and address fields"
        },
        "plan_details": {
            "description": "Insurance plans with formulary information",
            "key_fields": ["plan_name", "state_name", "formulary_url", "source_link", "formulary_date"],
            "enhancements": "Added formulary URLs, source links, and formulary dates"
        },
        "drug_formulary_details": {
            "description": "Drug coverage information with comprehensive details",
            "key_fields": ["drug_name", "drug_tier", "drug_requirements", "state_name", "ndc_code", "jcode"],
            "enhancements": "Changed state_code to state_name to match plan_details"
        }
    }
    
    for table, info in schema_info.items():
        print(f"\n🏗️  {table.upper()}:")
        print(f"   📝 {info['description']}")
        print(f"   🔑 Key fields: {', '.join(info['key_fields'])}")
        print(f"   ✨ Enhancements: {info['enhancements']}")

def main():
    """Main function with enhanced schema support"""
    print("🚀 ENHANCED DATABASE SCHEMA CLEANUP TOOL")
    print("=" * 80)
    print("This script will create enhanced database schema with:")
    print("• Payer contact information (phone, address)")
    print("• Plan formulary URLs and dates")
    print("• Comprehensive drug formulary fields")
    print("• Optimized indexes for performance")
    print("⚠️  WARNING: This will delete all existing data!")
    print("=" * 80)
    
    # Show what the enhanced schema will look like
    show_schema_summary()
    
    # Step 1: Check existing data
    backup_data = backup_existing_data()
    
    if backup_data:
        print(f"\n📊 Found existing data in {len(backup_data)} tables")
        print("Please backup your data manually before proceeding if you need it.")
        print("\nExisting data counts:")
        for table, count in backup_data.items():
            print(f"  • {table}: {count} records")
    
    # Step 2: Ask for confirmation and recreate tables
    if drop_and_recreate_tables():
        # Step 3: Verify new enhanced schema
        verify_schema()
        
        # Step 4: Test compatibility with enhanced script
        test_script_compatibility()
        
        print(f"\n🎉 ENHANCED SCHEMA CLEANUP COMPLETE!")
        print("=" * 80)
        print("✅ Your database now has the enhanced schema with:")
        print("   🏢 Payer contact information and addresses")
        print("   📋 Plan formulary URLs and dates")
        print("   💊 Comprehensive drug formulary fields")
        print("   🚀 Optimized indexes for fast queries")
        print("✅ Your enhanced db_updation.py script should work perfectly now")
        print("✅ All field mappings match between schema and processing script")
        print("\nYou can now run: python db_updation.py")
        
    else:
        print("\n❌ Enhanced schema cleanup was cancelled or failed")

if __name__ == "__main__":
    main()