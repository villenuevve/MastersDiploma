from sqlalchemy.orm import Session

from backend.database import Base, engine, SessionLocal
from backend.models import Role


def init_db():
    Base.metadata.create_all(bind=engine)

    # GROUND_TRUTH_MIGRATION
    with engine.begin() as connection:
        columns = {
            row[1]
            for row in connection.exec_driver_sql(
                "PRAGMA table_info(images)"
            ).fetchall()
        }

        migrations = {
            "has_ground_truth": (
                "ALTER TABLE images "
                "ADD COLUMN has_ground_truth BOOLEAN NOT NULL DEFAULT 0"
            ),
            "ground_truth_path": (
                "ALTER TABLE images "
                "ADD COLUMN ground_truth_path VARCHAR(500)"
            ),
            "ground_truth_format": (
                "ALTER TABLE images "
                "ADD COLUMN ground_truth_format VARCHAR(30)"
            )
        }

        for column_name, sql in migrations.items():
            if column_name not in columns:
                connection.exec_driver_sql(sql)
                print(f"Added images.{column_name}")

    # DATASET_SYSTEM_FLAG_MIGRATION
    with engine.begin() as connection:
        dataset_columns = {
            row[1]
            for row in connection.exec_driver_sql(
                "PRAGMA table_info(datasets)"
            ).fetchall()
        }
        if "is_system" not in dataset_columns:
            connection.exec_driver_sql(
                "ALTER TABLE datasets "
                "ADD COLUMN is_system BOOLEAN NOT NULL DEFAULT 0"
            )
            print("Added datasets.is_system")

        connection.exec_driver_sql(
            "UPDATE datasets SET is_system = 1 "
            "WHERE name = ? AND is_system = 0",
            ("Uploaded images",)
        )



    db: Session = SessionLocal()
    try:
        default_roles = ["User", "Administrator", "Technician"]

        for role_name in default_roles:
            existing_role = (
                db.query(Role)
                .filter(Role.name == role_name)
                .first()
            )

            if existing_role is None:
                db.add(Role(name=role_name))

        db.commit()
        print("Database initialized successfully.")
        print("Roles:", ", ".join(default_roles))

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
