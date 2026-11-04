# =============================================================================
# AgriLite-Hybrid Backend
# init_db.py - Database Initialization Script
# 
# Run this script to initialize the database with tables and seed data.
# Usage: python init_db.py
# =============================================================================

import os
import sys
import json

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from extensions import db, bcrypt
from models import User, Crop


def init_database():
    """
    Initialize the database with tables and seed data.
    
    Creates all tables and optionally seeds with initial data
    including default admin user and crop information.
    """
    app = create_app()
    
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✓ Database tables created successfully")
        
        # Check if admin user exists
        admin = User.query.filter_by(email='admin@agrilite.com').first()
        
        if not admin:
            # Create default admin user
            admin_password = bcrypt.generate_password_hash('Admin123!').decode('utf-8')
            admin = User(
                email='admin@agrilite.com',
                password_hash=admin_password,
                first_name='Admin',
                last_name='User',
                role='admin',
                is_active=True
            )
            db.session.add(admin)
            print("✓ Admin user created (email: admin@agrilite.com, password: Admin123!)")
        else:
            print("✓ Admin user already exists")
        
        # Seed crop data if not exists
        crops_data = [
            {
                'name': 'tomato',
                'display_name': 'Tomato',
                'description': 'Widely cultivated crop affected by various bacterial and fungal diseases.',
                'image_url': '/images/crops/tomato.jpg',
                'diseases': json.dumps([
                    'Bacterial Spot',
                    'Early Blight',
                    'Late Blight',
                    'Leaf Mold',
                    'Septoria Leaf Spot',
                    'Spider Mites (Two-spotted)',
                    'Target Spot',
                    'Yellow Leaf Curl Virus',
                    'Mosaic Virus',
                    'Healthy',
                    'Powdery Mildew'
                ])
            },
            {
                'name': 'chilli',
                'display_name': 'Chilli',
                'description': 'Spice crop prone to viral infections and fungal diseases.',
                'image_url': '/images/crops/chilli.jpg',
                'diseases': json.dumps([
                    'Anthracnose',
                    'Damping Off',
                    'Leaf Curl Virus',
                    'Leaf Spot',
                    'Veinal Mottle Virus',
                    'Whitefly',
                    'Yellowish',
                    'Healthy'
                ])
            }
        ]
        
        for crop_data in crops_data:
            existing_crop = Crop.query.filter_by(name=crop_data['name']).first()
            if not existing_crop:
                crop = Crop(**crop_data)
                db.session.add(crop)
                print(f"✓ Crop '{crop_data['display_name']}' added")
            else:
                print(f"✓ Crop '{crop_data['display_name']}' already exists")
        
        # Commit all changes
        db.session.commit()
        
        print("\n" + "="*50)
        print("Database initialization completed successfully!")
        print("="*50)
        print("\nDefault admin credentials:")
        print("  Email: admin@agrilite.com")
        print("  Password: Admin123!")
        print("\n⚠️  Please change the admin password in production!")


def reset_database():
    """
    Drop all tables and reinitialize the database.
    
    WARNING: This will delete all data!
    """
    app = create_app()
    
    with app.app_context():
        confirm = input("⚠️  This will DELETE ALL DATA. Type 'yes' to confirm: ")
        
        if confirm.lower() == 'yes':
            db.drop_all()
            print("✓ All tables dropped")
            
            db.create_all()
            print("✓ Tables recreated")
            
            init_database()
        else:
            print("Operation cancelled")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--reset':
        reset_database()
    else:
        init_database()
