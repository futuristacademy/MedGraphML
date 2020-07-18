import ijson

#print('Loaded local dev copy of autoTigerGraph!!')

def get_first(filename, n=1):

    with open(filename, 'r') as f:

        first = ['']*n
        objects = ijson.items(f, '', multiple_values=True, use_float=True)

        i=0
        for json_object in objects:
            first[i]=json_object
            i += 1
            if i >= n:
                return first

    return first[:i]


def vertex_from_json(json_object, vertex_name, fields=None):

    if fields == None:
        fields = list(json_object.keys())
    
    type_translate = {'str': 'STRING', 'dict': 'STRING', 'float': 'DOUBLE', 
                      'bool': 'BOOL', 'int': 'INT'}

    gsql_cmd  = 'VERTEX ' + vertex_name + ' (PRIMARY_ID ' 
    
    for field in [fields[0]]+fields:

        if field == 'date':
            gsql_cmd += 'date_time' + ' '
            gsql_cmd += 'DATETIME'
        else:    
            gsql_cmd += field + ' '
            gsql_cmd += type_translate[type(json_object[field]).__name__]
    
        gsql_cmd += ', '

    return gsql_cmd[:-2] + ')'

def problem_fields_to_str(json_object):

    for key, value in json_object.items():
        if isinstance(value, dict) or value == None:
            json_object[key] = str(value)
        if key == 'date':
            json_object.pop(key)
            json_object['date_time'] = value

    return json_object


def upsert_json_vertices(
    filename, 
    conn, 
    vertex_name, 
    primary_id, 
    field_paths=None,
    field_defaults=None,
    extract_func=None,
    batch_size=10000, 
    max_verts=10**10):

    ids = ['']*batch_size
    bodies = ['']*batch_size

    with open(filename, 'r') as f:
        objects = ijson.items(f, '', multiple_values=True, use_float=True)

        i = 0
        count = 0
        for json_object in objects:

            if json_object:

                if extract_func != None:
                   json_object = extract_func(json_object)

                elif field_paths != None:
                    json_object = extract_fields(json_object, 
                                                 field_paths, 
                                                 field_defaults)

                ids[i]=json_object[primary_id]

                bodies[i]=problem_fields_to_str(json_object)
                
                count += 1
                i += 1
                if count >= max_verts:
                    break
                if i%batch_size == 0:
                    conn.upsertVertices(vertex_name, list(zip(ids, bodies)))
                    i = 0

            else:
                break
                
    conn.upsertVertices(vertex_name, list(zip(ids[:i], bodies[:i])))

    return count

def upsert_json_edges(filename, 
                      conn, 
                      from_vertex, 
                      edge_name, 
                      to_vertex, 
                      from_field_path=None, 
                      to_field_path=None, 
                      extract_func=None, 
                      batch_size=10000,
                      max_edges=10**10, 
                      split_string=', '):

    froms = ['']*batch_size
    tos = ['']*batch_size

    field_paths={'from' : from_field_path,
                 'to': to_field_path}

    with open(filename, 'r') as f:
        objects = ijson.items(f, '', multiple_values=True, use_float=True)

        i = 0
        count = 0
        for json_object in objects:

            if json_object:

                if extract_func != None:
                    json_object = extract_func(json_object)
                else:
                    json_object = extract_fields(json_object, field_paths)
                
                json_froms = json_object['from'].split(split_string)
                json_tos = json_object['to'].split(split_string)
                
                if len(json_froms) > 1:
                    json_tos *= len(json_forms)
                elif len(json_tos) > 1:
                    json_froms *= len(json_tos)    
                
                for from_vert, to_vert in zip(json_froms, json_tos):

                    froms[i] = from_vert
                    tos[i] = to_vert
                    count += 1
                    i += 1
                    if count >= max_edges:
                        conn.upsertEdges(from_vertex, edge_name, to_vertex, list(zip(froms[:i], tos[:i])))
                        return count
                    if i%batch_size == 0:
                        conn.upsertEdges(from_vertex, edge_name, to_vertex, list(zip(froms, tos)))
                        i = 0

            else:
                break

    conn.upsertEdges(from_vertex, edge_name, to_vertex, list(zip(froms[:i], tos[:i])))
    return count

import ijson 
def extract_fields(json_object, field_paths, field_defaults=None):

    result={}
        
    for key, field_path in field_paths.items():
        result[key]=json_object
        
        for step in field_path:
            if ((isinstance(result[key], dict) 
                 and step in result[key])
                or 
                (isinstance(result[key], list)
                 and isinstance(step, int)
                 and step >=0 
                 and step < len(result[key]))):
            
                result[key] = result[key][step]
            else:
                result[key]=field_defaults[key]
            
    return result
