from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Boolean, Float, DateTime,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import relationship

from backend.database import Base


class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(30), unique=True, nullable=False)

    users = relationship("User", back_populates="role")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    role = relationship("Role", back_populates="users")
    datasets = relationship("Dataset", back_populates="owner")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), nullable=False)
    description = Column(String(500))
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="datasets")
    images = relationship("Image", back_populates="dataset")
    splits = relationship("DatasetSplit", back_populates="dataset")


class DatasetSplit(Base):
    __tablename__ = "dataset_splits"
    __table_args__ = (
        UniqueConstraint("dataset_id", "name", name="uq_dataset_split"),
    )

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    name = Column(String(20), nullable=False)

    dataset = relationship("Dataset", back_populates="splits")
    images = relationship("Image", back_populates="split")


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    split_id = Column(Integer, ForeignKey("dataset_splits.id"))
    uploaded_by = Column(Integer, ForeignKey("users.id"))

    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    file_size = Column(Integer)
    has_mask = Column(Boolean, default=False, nullable=False)
    mask_path = Column(String(500))

    has_ground_truth = Column(
        Boolean,
        default=False,
        nullable=False
    )
    ground_truth_path = Column(String(500))
    ground_truth_format = Column(String(30))
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="images")
    split = relationship("DatasetSplit", back_populates="images")
    predictions = relationship("Prediction", back_populates="image")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)

    model_name = Column(String(150))
    result_path = Column(String(500))
    inference_time_ms = Column(Float)
    mean_confidence = Column(Float)
    low_confidence_percentage = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    image = relationship("Image", back_populates="predictions")
    detected_classes = relationship(
        "DetectedClass",
        back_populates="prediction",
        cascade="all, delete-orphan"
    )


class DetectedClass(Base):
    __tablename__ = "detected_classes"
    __table_args__ = (
        UniqueConstraint(
            "prediction_id", "class_id",
            name="uq_prediction_class"
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(
        Integer, ForeignKey("predictions.id"), nullable=False
    )
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(100), nullable=False)
    pixel_count = Column(Integer, default=0)
    percentage = Column(Float)

    prediction = relationship(
        "Prediction",
        back_populates="detected_classes"
    )
