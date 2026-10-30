from datetime import datetime
from pydantic import BaseModel,Field,ConfigDict
class TelemetryEvent(BaseModel):
    model_config=ConfigDict(extra='forbid')
    timestamp:datetime; device_id:str=Field(min_length=1,max_length=100); device_model:str; os_version:str; region:str; app_version:str; release:str
    cpu_pct:float=Field(ge=0,le=100); gpu_pct:float=Field(ge=0,le=100); memory_mb:float=Field(ge=0); battery_drain_pct_hr:float=Field(ge=0); launch_time_ms:float=Field(ge=0); fps:float=Field(ge=0,le=120); thermal_c:float; network_latency_ms:float=Field(ge=0)
class AskRequest(BaseModel): question:str=Field(min_length=3,max_length=1000)
