
import autoTigerGraph as atg

def uploadAll(conn, data_dir = 'output/fhir'):

    uploadConditions(conn, data_dir=data_dir)
    uploadPatients(conn, data_dir=data_dir)
    uploadEncounters(conn, data_dir=data_dir)
    uploadProviders(conn, data_dir=data_dir)
    uploadProcedures(conn, data_dir=data_dir)
    uploadMedication(conn, data_dir=data_dir)

def uploadConditions(conn, data_dir='output/fhir'):

    filename = data_dir + '/Condition.ndjson'

    field_paths={
        'condition_id': ['id'],
        'description': ['code', 'coding', 0, 'display'],
        'startDate': ['onsetDateTime'],
        'endDate': ['abatementDateTime'] 
    }

    field_defaults = {
        'endDate': '3000-01-01T00:00:00+00:00'
    }

    vertex_name = 'Condition'
    primary_id = 'condition_id'

    atg.upsert_json_vertices(
        filename=filename, 
        conn=conn, 
        vertex_name=vertex_name, 
        primary_id=primary_id, 
        field_paths=field_paths,
        field_defaults=field_defaults)

    def extract_patient_condition(json_object):

        field_paths = {
            'from': ['id'],
            'to': ['subject','reference'],
        }

        json_object = atg.extract_fields(
            json_object=json_object, field_paths=field_paths)
        
        json_object = {
            'from': json_object['from'],
            'to': json_object['to'].replace('Patient/','')
        }
        
        return json_object

    atg.upsert_json_edges(
        filename=filename, 
        conn=conn, 
        from_vertex='Condition', 
        edge_name='PATIENT_HAS_CONDITION', 
        to_vertex='Patient', 
        extract_func=extract_patient_condition)
    
    def extract_condition_encounter(json_object):

        field_paths = {
            'from': ['id'],
            'to': ['encounter','reference'],
        }

        json_object = atg.extract_fields(
            json_object=json_object, field_paths=field_paths)
        
        json_object = {
            'from': json_object['from'],
            'to': json_object['to'].replace('Encounter/','')
        }
        
        return json_object

    atg.upsert_json_edges(
        filename=filename, 
        conn=conn, 
        from_vertex='Condition', 
        edge_name='ENCOUNTER_FOR_CONDITION', 
        to_vertex='Encounter', 
        extract_func=extract_condition_encounter)

def uploadPatients(conn, data_dir = 'output/fhir'):

    filename = data_dir + '/Patient.ndjson'

    def extract_patient(json_object):

        field_paths={
            'patient_id': ['id'],
            'first_name': ['name',0,'given',0],
            'last_name': ['name',0,'family'],
            'gender': ['gender'],
            'birth': ['birthDate'],
            'death': ['deceasedDateTime'],
        }

        field_defaults = {
            'death': '3000-01-01T00:00:00+00:00',
        }

        json_object = atg.extract_fields(
            json_object=json_object, 
            field_paths=field_paths, 
            field_defaults=field_defaults)
        
        json_object['name'] =  remove_digits(
            json_object['first_name'] + ' ' + json_object['last_name'])
        
        json_object.pop('first_name')
        json_object.pop('last_name')
        
        return json_object

    vertex_name = 'Patient'
    primary_id = 'patient_id'

    atg.upsert_json_vertices(
        filename=filename, 
        conn=conn, 
        vertex_name=vertex_name, 
        primary_id=primary_id, 
        extract_func=extract_patient)


def uploadEncounters(conn, data_dir = 'output/fhir'):

    filename = data_dir + '/Encounter.ndjson'

    field_paths={
        'encounter_id': ['id'],
        'description': ['type', 0, 'text'],
        'startTime': ['period','start'],
        'endTime': ['period','end']
        }

    vertex_name = 'Encounter'
    primary_id = 'encounter_id'

    atg.upsert_json_vertices(
        filename=filename, 
        conn=conn, 
        vertex_name=vertex_name, 
        primary_id=primary_id, 
        field_paths=field_paths)
    
    def extract_encounter_practitioner(json_object):

        field_paths={
            'from': ['id'],
            'to': ['participant',0,'individual','reference'],
        }

        json_object = atg.extract_fields(
            json_object=json_object, field_paths=field_paths)
        
        json_object = {
            'from': json_object['from'],
            'to': json_object['to'].replace('Practitioner/','')
        }
        
        return json_object

    atg.upsert_json_edges(
        filename=filename, 
        conn=conn, 
        from_vertex='Encounter', 
        edge_name='ENCOUNTER_HAS_PROVIDER', 
        to_vertex='Provider', 
        extract_func=extract_encounter_practitioner)
    
    def extract_encounter_subject(json_object):

        field_paths={
            'from': ['id'],
            'to': ['subject','reference'],
        }

        json_object = atg.extract_fields(
            json_object=json_object, field_paths=field_paths)
        
        json_object = {
            'from': json_object['from'],
            'to': json_object['to'] .replace('Patient/','')
        }
        
        return json_object

    atg.upsert_json_edges(
        filename=filename, 
        conn=conn, 
        from_vertex='Encounter', 
        edge_name='ENCOUNTER_FOR_PATIENT', 
        to_vertex='Patient', 
        extract_func=extract_encounter_subject)


def uploadProviders(conn, data_dir = 'output/fhir'):

    filename = data_dir + '/PractitionerRole.ndjson'

    def extract_practitioner(json_object):

        field_paths={
            'provider_id': ['practitioner','reference'],
            'name': ['practitioner','display'],
            'specialty': ['specialty',0,'text']
        }

        json_object = atg.extract_fields(
            json_object=json_object, field_paths=field_paths)
        
        json_object['provider_id'] = json_object['provider_id'].replace(
            'Practitioner/','')
        
        json_object['name'] = remove_digits(json_object['name'])
        
        return json_object

    vertex_name = 'Provider'
    primary_id = 'provider_id'

    atg.upsert_json_vertices(
        filename=filename, 
        conn=conn, 
        vertex_name=vertex_name, 
        primary_id=primary_id, 
        extract_func=extract_practitioner)
    

def uploadProcedures(conn, data_dir = 'output/fhir'):

    filename = data_dir + '/Procedure.ndjson'
    
    field_paths={
        'procedure_id': ['id'],
        'description': ['code','text'],
        'startTime': ['performedPeriod','start'],
        'endTime': ['performedPeriod','end']
    }

    vertex_name = 'Procedures'
    primary_id = 'procedure_id'

    atg.upsert_json_vertices(
        filename=filename, 
        conn=conn, 
        vertex_name=vertex_name, 
        primary_id=primary_id, 
        field_paths=field_paths)

    def extract_procedure_subject(json_object):

        field_paths={
            'from': ['id'],
            'to': ['subject','reference'],
        }

        json_object = atg.extract_fields(
            json_object=json_object, field_paths=field_paths)
        
        json_object = {
            'from': json_object['from'],
            'to': json_object['to'].replace('Patient/','')
        }
        
        return json_object

    atg.upsert_json_edges(
        filename=filename, 
        conn=conn, 
        from_vertex='Procedures', 
        edge_name='PATIENT_HAS_PROCEDURE', 
        to_vertex='Patient', 
        extract_func=extract_procedure_subject)
    
    def extract_procedure_encounter(json_object):

        field_paths={
            'from': ['id'],
            'to': ['encounter','reference'],
        }

        json_object = atg.extract_fields(
            json_object=json_object, field_paths=field_paths)
        
        json_object = {
            'from': json_object['from'],
            'to': json_object['to'].replace('Encounter/','')
        }
        
        return json_object

    atg.upsert_json_edges(
        filename=filename, 
        conn=conn, 
        from_vertex='Procedures', 
        edge_name='ENCOUNTER_FOR_PROCEDURE', 
        to_vertex='Encounter', 
        extract_func=extract_procedure_encounter)
    
def uploadMedication(conn, data_dir = 'output/fhir'):

    filename = data_dir + '/MedicationRequest.ndjson'
    
    field_paths={
        'medication_id': ['id'],
        'description': ['medicationCodeableConcept','text'],
        'authorDate': ['authoredOn'],
    }

    field_defaults={
        'description': '',
    }

    vertex_name = 'Medication'
    primary_id = 'medication_id'

    atg.upsert_json_vertices(
        filename=filename, 
        conn=conn, 
        vertex_name=vertex_name, 
        primary_id=primary_id, 
        field_paths=field_paths,
        field_defaults=field_defaults)
    
    def extract_medication_subject(json_object):

        field_paths={
            'from': ['id'],
            'to': ['subject','reference'],
        }

        json_object = atg.extract_fields(
            json_object=json_object, field_paths=field_paths)
        
        json_object = {
            'from': json_object['from'],
            'to': json_object['to'].replace('Patient/','')
        }
        
        return json_object

    atg.upsert_json_edges(
        filename=filename, 
        conn=conn, 
        from_vertex='Medication', 
        edge_name='PATIENT_HAS_MEDICATION', 
        to_vertex='Patient', 
        extract_func=extract_medication_subject)
    
    def extract_medication_encounter(json_object):

        field_paths={
            'from': ['id'],
            'to': ['encounter','reference'],
        }

        json_object = atg.extract_fields(
            json_object=json_object, field_paths=field_paths)
        
        json_object = {
            'from': json_object['from'],
            'to': json_object['to'].replace('Encounter/','')
        }
        
        return json_object

    atg.upsert_json_edges(
        filename=filename, 
        conn=conn, 
        from_vertex='Medication', 
        edge_name='ENCOUNTER_FOR_MEDICATION', 
        to_vertex='Encounter', 
        extract_func=extract_medication_encounter)


def remove_digits(s):
    return ''.join(i for i in s if not i.isdigit())
