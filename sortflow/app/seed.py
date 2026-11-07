from sqlalchemy.orm import Session
from app.models import SortCenter

CENTERS = [
    ("SC-01","Northeast-1","Northeast",18000,14200,42,38),
    ("SC-02","Northeast-2","Northeast",16000,15100,38,40),
    ("SC-03","Mid-Atlantic-1","Mid-Atlantic",14000,11800,34,32),
    ("SC-04","Mid-Atlantic-2","Mid-Atlantic",15500,13200,36,35),
    ("SC-05","Southeast-1","Southeast",17000,12500,40,34),
    ("SC-06","Southeast-2","Southeast",13000,10900,30,29),
    ("SC-07","Midwest-1","Midwest",19000,16000,45,42),
    ("SC-08","Midwest-2","Midwest",15000,14300,35,38),
    ("SC-09","West-1","West",21000,17200,48,43),
    ("SC-10","West-2","West",17500,13900,41,36),
    ("SC-11","Central-1","Central",14500,10200,33,30),
    ("SC-12","Central-2","Central",16500,12800,37,34),
]

def seed_centers(db: Session):
    if db.query(SortCenter).count():
        return
    for cid, name, region, cap, demand, workers, required in CENTERS:
        db.add(SortCenter(
            id=cid, name=name, region=region,
            capacity_per_hour=cap, current_demand_per_hour=demand,
            available_workers=workers, workers_required=required,
            processing_cost_per_package=round(0.075 + (int(cid[-2:]) % 5) * 0.008, 3),
        ))
    db.commit()
