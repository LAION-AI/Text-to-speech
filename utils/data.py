import json
import shortuuid
import pandas as pd

def normalize_json(obj, parent_key=None):
    updated_data = {}
    
    def _normalize(obj, parent_key):
        nonlocal updated_data
        for key, value in obj.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict):
                _normalize(value, new_key)
            elif isinstance(value, list):
                if not value or not isinstance(value[0], dict):
                    updated_data[new_key] = json.dumps(value)
                else:
                    _normalize(value[0], new_key)
            else:
                updated_data[new_key] = value

    if isinstance(obj, list):
        for item in obj:
            _normalize(item, parent_key)
    else:
        _normalize(obj, parent_key)
    
    if isinstance(updated_data, dict):
        records, meta = [], []
        for k, v in updated_data.items():
            if isinstance(v, list):
                records.append(k)
            else:
                meta.append(k)
        
        dfs = [
            pd.json_normalize(
                updated_data,
                record_path=record,
                record_prefix=f"{record}.",
                meta=meta,
            )
            for record in records
        ]
        
        df = pd.concat(dfs, ignore_index=True)
        index = [str(shortuuid.uuid()) for _ in range(df.shape[0])]
        return df.rename(index=dict(zip(list(df.index), index)))

    return updated_data