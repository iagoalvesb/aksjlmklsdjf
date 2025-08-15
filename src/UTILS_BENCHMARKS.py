class assin2rte:
    def __init__(self):
        self.dataset_path = "eduagarcia/portuguese_benchmark"
        self.subset = "assin2-rte"
        self.split = "test"
        self.important_columns = ["idx", "sentence1", "sentence2", "label"]
        self.label_column = "label"
        self.answer_pattern = "yes_no"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Premissa: {example['sentence1']}.\nHipótese: {example['sentence2']}.\n\nCom base na nossa premissa, a hipótese acima é verdadeira ou falsa? Responda com Sim ou Não."

        assistant_message_positive = "Resposta: Sim, com base na nossa premissa, a hipótese acima é verdadeira."
        assistant_message_negative = "Resposta: Não, com base na nossa premissa, a hipótese acima não é verdadeira."
        prompt_informations['assistant_message_with_answer'] = assistant_message_positive if example['label'] == '1' else assistant_message_negative
        
        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations


class assin2sts:
    def __init__(self):
        self.dataset_path = "eduagarcia/portuguese_benchmark"
        self.subset = "assin2-sts"
        self.split = "test"
        self.important_columns = ["idx", "sentence1", "sentence2", "label"]
        self.label_column = "label"
        self.answer_pattern = "continue_value"
        self.answer_type = "continue"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Sentença 1: {example['sentence1']}.\nSentença 2: {example['sentence2']}.\n\nApós ler as sentenças, o quão similares elas são? Utilize valores entre 1 e 5, sendo 1 pouco similares e 5 muito similares."
        prompt_informations['assistant_message_with_answer'] = f"Resposta: Ao ler a Sentença 1 e a Sentença 2, o valor da similaridade é de {example['label']}, ou seja, são {'pouco' if float(example['label']) < 3 else 'muito'} similares."
        prompt_informations['assistant_message_without_answer'] = "Resposta: Ao ler a Sentença 1 e a Sentença 2, o valor da similaridade é de "
        return prompt_informations


class bluex:
    def __init__(self):
        self.dataset_path = "eduagarcia-temp/BLUEX_without_images"
        self.subset = None
        self.split = "train"
        self.important_columns = ["question", "choices", "answerKey"]
        self.label_column = "answerKey"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        num_alternatives_example = len(example['choices']['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {example['choices']['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)
        
        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"
        
        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations


class enem:
    def __init__(self):
        self.dataset_path = "eduagarcia/enem_challenge"
        self.subset = None
        self.split = "train"
        self.important_columns = ["question", "choices", "answerKey"]
        self.label_column = "answerKey"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        num_alternatives_example = len(example['choices']['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {example['choices']['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)
        
        prompt_informations['user_message'] = f"Pergunta:\n{example['question']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"
        
        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations


class hatebr:
    def __init__(self):
        self.dataset_path = "eduagarcia/portuguese_benchmark"
        self.subset = "HateBR_offensive_binary"
        self.split = "test"
        self.important_columns = ["sentence", "label"]
        self.label_column = "label"
        self.answer_pattern = "yes_no"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Comentário: {example['sentence']}.\nO comentário acima é ofensivo? Responda com Sim ou Não."""

        assistant_message_positive = "Resposta: Sim, o texto é ofensivo."
        assistant_message_negative = "Resposta: Não, o texto não é ofensivo."
        prompt_informations['assistant_message_with_answer'] = assistant_message_positive if example['label'] == '1' else assistant_message_negative
        
        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations


class portuguese_hate_speech:
    def __init__(self):
        self.dataset_path = "eduagarcia/portuguese_benchmark"
        self.subset = "Portuguese_Hate_Speech_binary"
        self.split = "test"
        self.important_columns = ["sentence", "label"]
        self.label_column = "label"
        self.answer_pattern = "yes_no"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Comentário: {example['sentence']}.\nO comentário acima é ofensivo? Responda com Sim ou Não."""

        assistant_message_positive = "Resposta: Sim, o texto é ofensivo."
        assistant_message_negative = "Resposta: Não, o texto não é ofensivo."
        prompt_informations['assistant_message_with_answer'] = assistant_message_positive if example['label'] == '1' else assistant_message_negative
        
        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations


class toxsyn_pt:
    def __init__(self):
        self.dataset_path = "ToxSyn/ToxSyn-PT"
        self.subset = None
        self.split = "test"
        self.important_columns = ["text", "is_toxic"]
        self.label_column = "is_toxic"
        self.answer_pattern = "yes_no"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Comentário: {example['text']}.\nO comentário acima é ofensivo? Responda com Sim ou Não."""

        assistant_message_positive = "Resposta: Sim, o texto é ofensivo."
        assistant_message_negative = "Resposta: Não, o texto não é ofensivo."
        prompt_informations['assistant_message_with_answer'] = assistant_message_positive if example['label'] == '1' else assistant_message_negative
        
        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations


class faquad:
    def __init__(self):
        self.dataset_path = "ruanchaves/faquad-nli"
        self.subset = None
        self.split = "test"
        self.important_columns = ["question", "answer", "label"]
        self.label_column = "label"
        self.answer_pattern = "yes_no"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Pergunta: {example['question']}\n\nResposta: {example['answer']}\nA resposta dada satisfaz à pergunta? Responda com Sim ou Não."""

        assistant_message_positive = "Resposta: Sim, a resposta dada satisfaz à pergunta."
        assistant_message_negative = "Resposta: Não, a resposta dada não satisfaz à pergunta."
        prompt_informations['assistant_message_with_answer'] = assistant_message_positive if example['label'] == 1 else assistant_message_negative
        
        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations

class tweetsentbr:
    def __init__(self):
        self.dataset_path = "EdwardSJ151/tweetsentbr_fewshot_ptlabel"
        self.subset = None
        self.split = "test"
        self.important_columns = ["sentence", "label"]
        self.label_column = "label"
        self.answer_pattern = "multiple_choice_full_word"
        self.answer_type = "category"   

    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        prompt_informations['user_message'] = f"Comentário: {example['sentence']}.\nQual é o tom do comentário acima? Responda com apenas uma das seguintes opções: Positivo, Negativo ou Neutro."""

        label_map = {
            "Positivo": "Resposta: Positivo",
            "Negativo": "Resposta: Negativo",
            "Neutro": "Resposta: Neutro"
        }

        prompt_informations['assistant_message_with_answer'] = label_map.get(example['label']) 
        
        prompt_informations['assistant_message_without_answer'] = "Resposta: "
        return prompt_informations
    

class oab:
    def __init__(self):
        self.dataset_path = "cemig-ceia/vestibulares_concatened"
        self.subset = None
        self.split = "oab"
        self.important_columns = ["statement", "alternatives", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."

        alternatives_dict = ast.literal_eval(example['alternatives'])

        num_alternatives_example = len(alternatives_dict['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {alternatives_dict['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)
        
        prompt_informations['user_message'] = f"Pergunta:\n{example['statement']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"
        
        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations

class poscomp:
    def __init__(self):
        self.dataset_path = "cemig-ceia/vestibulares_concatened"
        self.subset = None
        self.split = "poscomp"
        self.important_columns = ["statement", "alternatives", "answer"]
        self.label_column = "answer"
        self.answer_pattern = "multiple_choice"
        self.answer_type = "category"


    def get_prompt_informations(self, example):
        prompt_informations = {}
        prompt_informations['base_system_message'] = "Você é um assistente prestativo, responda de forma direta e objetiva."
        
        alternatives_dict = ast.literal_eval(example['alternatives'])

        num_alternatives_example = len(alternatives_dict['text'])
        example_alternatives = [f"({chr(65 + asc_letter)}): {alternatives_dict['text'][asc_letter]}" for asc_letter in range(num_alternatives_example)]
        example_alternatives = "\n".join(example_alternatives)
        
        
        prompt_informations['user_message'] = f"Pergunta:\n{example['statement']}.\n\nLeia as alternativas abaixo e responda corretamente a pergunta:\n\n{example_alternatives}"

        prompt_informations['assistant_message_with_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ({example['label']})"
        
        prompt_informations['assistant_message_without_answer'] = f"Lendo as alternativas {''.join(('(' + chr(65 + asc_letter) + '),' for asc_letter in range(num_alternatives_example)))} a alternativa que responde corretamente a pergunta é a letra ("
        return prompt_informations




BENCHMARKS_INFORMATIONS = {
    "assin2rte": assin2rte(),
    "assin2sts": assin2sts(),
    "bluex": bluex(),
    "enem": enem(),
    "hatebr": hatebr(),
    "portuguese_hate_speech": portuguese_hate_speech(),
    "toxsyn_pt": toxsyn_pt(),
    "faquad": faquad(),
    "tweetsentbr": tweetsentbr(),
    "oab": oab(),
    "poscomp": poscomp(),
    }

