"""
Inferência do tipo de carta a partir do nome digitado pelo usuário.

A ideia: o nome de uma carta costuma dar pistas sobre o que ela é. Um verbo
ou uma ação ("Destroy", "Draw a Card") sugere instantânea/feitiço; um lugar
("Forest", "Mountain Pass") sugere terreno; uma profissão ou nome próprio
("Wizard", "Serra Angel", "Chandra") sugere criatura; um objeto ("Sword",
"Mind Stone") sugere artefato; e conceitos abstratos ("Pacifism", "Fear")
sugerem encantamento. Quando nada disso é claro (ou há empate), devolvemos
None e o pipeline sorteia um tipo.

Como funciona
=============
1. O nome é tokenizado e passa por POS tagging do NLTK (para achar verbos
   em nomes como "Draw a Card").
2. Cada palavra recebe "votos" de tipo, vindos de duas fontes:
   - listas curadas (inglês + português) de lugares, profissões/criaturas,
     objetos, conceitos abstratos e ações -- cobrem jargão de Magic e
     palavras que o WordNet classifica de forma enganosa;
   - WordNet (`synset.lexname()`), que mapeia categorias como
     noun.location -> terreno, noun.person -> criatura,
     noun.artifact -> artefato, noun.cognition -> encantamento etc.
3. A última palavra "principal" do nome (a cabeça do sintagma, antes de
   "of"/"de") pesa mais que as outras. Palavras curadas também pesam mais,
   para conseguirmos tratar "Tower"/"Temple" como lugar (terreno) mesmo o
   WordNet classificando como artefato.
4. Se nenhum tipo tiver sinal suficiente, tentamos a regra de nome próprio:
   uma palavra capitalizada, sem dígitos e com forma plausível de nome
   (ex: "Vizzerdrix", "Chandra") -> criatura.
5. Empate ou nenhum sinal reconhecível -> None (o `generator` sorteia).

Limitações (documentadas de propósito)
======================================
É uma heurística, não um classificador treinado: nomes ambíguos podem cair
no tipo "errado" do ponto de vista do Magic real (ex: "Day of Judgment"
pode virar instantânea em vez de feitiço). O WordNet é inglês; o suporte a
português é feito por listas curadas de termos comuns. Nomes inventados
sem nenhuma palavra conhecida são tratados como nome próprio (criatura) ou
como indeterminados, dependendo da forma.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn

from .tipos import TIPOS_DISPLAY

# -- Recursos do NLTK -------------------------------------------------

def _garantir_recursos() -> None:
    """Garante os corpora/modelos do NLTK usados aqui (idempotente)."""
    for recurso in (
        "corpora/wordnet",
        "tokenizers/punkt_tab",
        "taggers/averaged_perceptron_tagger_eng",
    ):
        try:
            nltk.data.find(recurso)
        except LookupError:
            nltk.download(recurso.split("/")[-1], quiet=True)


# -- Normalização de palavras ----------------------------------------

def _sem_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalizar(palavra: str) -> str:
    return _sem_acentos(palavra).casefold()


STOPWORDS: Set[str] = {
    # inglês
    "the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for",
    "from", "with", "by", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "my", "your",
    "his", "her", "their", "our", "no", "not", "nor", "but", "if", "then",
    "than", "so", "do", "does", "did", "has", "have", "had", "will",
    "would", "can", "could", "should", "may", "might", "must",
    # português (o suficiente para não virar "cabeça" do nome)
    "o", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "e", "ou", "em", "no", "na", "nos", "nas", "por", "para", "com", "sem",
    "ao", "aos", "que", "se", "como", "mais", "menos", "muito", "muita",
}

PREPOSICOES: Set[str] = {"of", "de", "do", "da", "dos", "das"}


# -- Listas curadas ---------------------------------------------------
# Termos em inglês (língua do oracle_text) e português (interface). As
# listas são normalizadas (sem acento, minúsculas) na hora da comparação.

LUGARES: Set[str] = {
    # inglês
    "forest", "mountain", "island", "swamp", "plains", "plain", "jungle",
    "desert", "oasis", "valley", "vale", "canyon", "gorge", "cave",
    "cavern", "grotto", "tunnel", "mine", "quarry", "pit", "abyss",
    "chasm", "river", "lake", "sea", "ocean", "coast", "shore", "beach",
    "bay", "gulf", "harbor", "harbour", "port", "dock", "marsh", "bog",
    "fen", "moor", "glade", "grove", "woods", "wood", "thicket", "garden",
    "meadow", "field", "farm", "mill", "village", "town", "city",
    "citadel", "tower", "temple", "shrine", "chapel", "abbey", "monastery",
    "academy", "school", "library", "archives", "observatory", "palace",
    "castle", "keep", "fort", "fortress", "outpost", "camp", "colony",
    "settlement", "province", "territory", "realm", "domain", "kingdom",
    "empire", "ruin", "ruins", "tomb", "crypt", "catacomb", "graveyard",
    "cemetery", "battlefield", "wasteland", "badlands", "highland",
    "highlands", "lowland", "lowlands", "ridge", "peak", "summit", "cliff",
    "waterfall", "spring", "reef", "islet", "peninsula", "steppe",
    "savanna", "prairie", "tundra", "glacier", "volcano", "crater",
    "sanctuary", "refuge", "haven", "lighthouse", "bridge", "road",
    "pass", "gate", "crossroads", "plaza", "market", "bazaar", "arena",
    "stadium", "theater", "theatre", "prison", "dungeon",
    # português
    "floresta", "montanha", "ilha", "pantano", "planicie", "planicies",
    "deserto", "oasis", "vale", "caverna", "gruta", "tunel", "mina",
    "pedreira", "poco", "abismo", "rio", "lago", "mar", "oceano", "costa",
    "praia", "baia", "golfo", "porto", "doca", "charco", "pantano",
    "campo", "fazenda", "moinho", "vila", "cidade", "cidadela", "torre",
    "templo", "santuario", "capela", "abadia", "mosteiro", "academia",
    "escola", "biblioteca", "arquivos", "observatorio", "palacio",
    "castelo", "forte", "fortaleza", "posto", "acampamento", "colonia",
    "assentamento", "provincia", "territorio", "reino", "dominio",
    "imperio", "ruina", "ruinas", "tumulo", "cripta", "catacumba",
    "cemitario", "campo de batalha", "deserto", "montanha", "colina",
    "colinas", "planicie", "pico", "summito", "penhasco", "cachoeira",
    "nascente", "recife", "ilheu", "peninsula", "estepe", "savana",
    "pradaria", "tundra", "geleira", "vulcao", "cratera", "refugio",
    "abrigo", "farol", "ponte", "estrada", "passagem", "portao",
    "encruzilhada", "praca", "mercado", "bazar", "arena", "estadio",
    "teatro", "prisao", "calabouco",
}

CRIATURAS: Set[str] = {
    # profissões
    "wizard", "mage", "sorcerer", "witch", "warlock", "cleric", "priest",
    "priestess", "paladin", "knight", "soldier", "warrior", "barbarian",
    "rogue", "thief", "assassin", "ranger", "scout", "hunter", "tracker",
    "druid", "shaman", "monk", "ninja", "samurai", "pirate", "mercenary",
    "captain", "general", "commander", "king", "queen", "prince",
    "princess", "emperor", "empress", "lord", "lady", "baron", "duke",
    "duchess", "chief", "elder", "sage", "oracle", "prophet", "seer",
    "mystic", "healer", "doctor", "alchemist", "artificer", "engineer",
    "smith", "bard", "minstrel", "scholar", "teacher", "guardian",
    "sentinel", "warden", "keeper", "champion", "hero", "villain",
    "wanderer", "nomad", "pilgrim", "explorer", "adventurer", "slayer",
    "reaper", "bringer", "herald", "envoy", "ambassador", "judge",
    "inquisitor", "templar", "crusader", "bishop", "abbot", "sculptor",
    "artist", "painter", "poet", "writer", "musician", "dancer", "actor",
    "chef", "cook",
    # criaturas fantásticas
    "angel", "demon", "devil", "dragon", "goblin", "elf", "dwarf", "orc",
    "troll", "ogre", "giant", "titan", "elemental", "spirit", "ghost",
    "phantom", "specter", "wraith", "zombie", "vampire", "skeleton",
    "lich", "werewolf", "wolf", "bear", "cat", "lion", "tiger", "panther",
    "leopard", "jaguar", "lynx", "fox", "dog", "hound", "boar", "stag",
    "deer", "elk", "horse", "pony", "unicorn", "pegasus", "griffin",
    "gryphon", "phoenix", "hydra", "serpent", "snake", "viper", "cobra",
    "spider", "scorpion", "insect", "beetle", "ant", "bee", "wasp",
    "bird", "eagle", "hawk", "falcon", "owl", "raven", "crow", "vulture",
    "bat", "rat", "mouse", "rabbit", "hare", "frog", "toad", "lizard",
    "gecko", "turtle", "tortoise", "crab", "fish", "shark", "whale",
    "dolphin", "octopus", "squid", "kraken", "leviathan", "merfolk",
    "mermaid", "merman", "siren", "nymph", "dryad", "treant", "ent",
    "treefolk", "beast", "monster", "horror", "nightmare", "fiend", "imp",
    "gremlin", "gargoyle", "golem", "construct", "automaton", "robot",
    "sphinx", "chimera", "manticore", "basilisk", "wyvern", "wurm",
    "worm", "sliver", "eldrazi", "myr", "thopter", "scarecrow", "avatar",
    "incarnation", "god", "goddess", "deity",
    # português
    "mago", "maga", "feiticeiro", "feiticeira", "bruxo", "bruxa",
    "clerigo", "sacerdote", "paladino", "cavaleiro", "soldado",
    "guerreiro", "barbaro", "ladrao", "assassino", "patrulheiro",
    "cacador", "druida", "xama", "monge", "ninja", "samurai", "pirata",
    "mercenario", "capitao", "general", "comandante", "rei", "rainha",
    "principe", "princesa", "imperador", "imperatriz", "senhor",
    "senhora", "baron", "duque", "duquesa", "chefe", "anciao", "sabio",
    "oraculo", "profeta", "vidente", "curandeiro", "alquimista",
    "artifice", "engenheiro", "ferreiro", "bardo", "menestrel",
    "estudioso", "guardiao", "sentinela", "campeao", "heroi", "andarilho",
    "nomade", "peregrino", "explorador", "aventureiro", "escultor",
    "artista", "pintor", "poeta", "escritor", "musico", "dancarino",
    "ator", "cozinheiro", "anjo", "demonio",
    "diabo", "dragao", "goblin", "elfo", "anao", "orc", "troll", "ogro",
    "gigante", "tita", "elemental", "espirito", "fantasma", "espectro",
    "zumbi", "vampiro", "esqueleto", "lobisomem", "lobo", "urso", "gato",
    "leao", "tigre", "pantera", "raposa", "cao", "javali", "cervo",
    "cavalo", "unicornio", "pegaso", "grifo", "fenix", "hidra",
    "serpente", "cobra", "aranha", "escorpiao", "inseto", "abelha",
    "passaro", "aguia", "falcao", "coruja", "corvo", "morcego", "rato",
    "coelho", "sapo", "lagarto", "tartaruga", "caranguejo", "peixe",
    "tubarao", "baleia", "golfinho", "polvo", "kraken", "sereia",
    "ninfa", "driade", "ente", "fera", "monstro", "pesadelo", "golem",
    "constructo", "automato", "esfinge", "quimera", "basilisco",
    "verme", "deus", "deusa",
}

OBJETOS: Set[str] = {
    # inglês
    "sword", "blade", "axe", "hammer", "mace", "spear", "lance", "bow",
    "crossbow", "arrow", "dagger", "knife", "shield", "armor", "armour",
    "helm", "helmet", "gauntlet", "glove", "boot", "cloak", "robe", "belt",
    "ring", "amulet", "talisman", "pendant", "necklace", "crown",
    "scepter", "sceptre", "orb", "sphere", "crystal", "gem", "jewel",
    "stone", "rock", "idol", "totem", "statue", "figurine", "chalice",
    "grail", "cup", "goblet", "vessel", "cauldron", "potion", "elixir",
    "brew", "scroll", "tome", "book", "grimoire", "map", "compass", "key",
    "lock", "lantern", "lamp", "candle", "mirror", "lens", "mask",
    "banner", "flag", "standard", "sigil", "emblem", "medallion", "coin",
    "treasure", "chest", "bag", "sack", "vial", "flask", "bottle", "jar",
    "box", "cage", "chain", "rope", "net", "trap", "bomb", "cannon",
    "ballista", "catapult", "engine", "machine", "device", "artifact",
    "relic", "staff", "wand", "rod", "tablet", "monolith", "obelisk",
    "pillar", "column", "throne", "altar", "forge", "anvil", "workshop",
    "laboratory", "ship", "boat", "canoe", "raft", "cart", "wagon",
    "carriage", "chariot", "lotus",
    # português
    "espada", "lamina", "machado", "martelo", "maca", "lanca", "arco",
    "besta", "flecha", "adaga", "faca", "escudo", "armadura", "elmo",
    "manopla", "luva", "bota", "capa", "manto", "cinto", "anel",
    "amuleto", "talisma", "pingente", "colar", "coroa", "cetro", "orbe",
    "esfera", "cristal", "gema", "joia", "pedra", "rocha", "idolo",
    "totem", "estatua", "estatueta", "calice", "graal", "taca", "vaso",
    "caldeirao", "pocao", "elixir", "pergaminho", "tomo", "livro",
    "grimorio", "mapa", "bussola", "chave", "fechadura", "lanterna",
    "lampada", "vela", "espelho", "lente", "mascara", "estandarte",
    "bandeira", "simbolo", "emblema", "medalhao", "moeda", "tesouro",
    "bau", "bolsa", "saco", "frasco", "garrafa", "jarro", "caixa",
    "gaiola", "corrente", "corda", "rede", "armadilha", "bomba", "canhao",
    "balista", "catapulta", "motor", "maquina", "dispositivo", "artefato",
    "reliquia", "cajado", "varinha", "bastao", "monolito", "obelisco",
    "pilar", "coluna", "trono", "altar", "forja", "bigorna", "oficina",
    "laboratorio", "navio", "barco", "canoa", "jangada", "carroca",
    "carruagem", "lotus",
}

ENCANTAMENTOS: Set[str] = {
    # inglês
    "pacifism", "faith", "courage", "honor", "honour", "glory", "destiny",
    "fate", "oath", "vow", "promise", "curse", "blessing", "grace",
    "mercy", "hope", "despair", "fear", "greed", "rage", "anger", "love",
    "hate", "peace", "harmony", "discord", "order", "chaos", "law",
    "justice", "judgment", "judgement", "truth", "lie", "dream", "memory",
    "thought", "mind", "aura", "enchantment", "hex", "jinx", "doom",
    "gloom", "blight", "decay", "corruption", "purity", "virtue", "sin",
    "guilt", "sorrow", "joy", "passion", "desire", "ambition", "pride",
    "envy", "silence", "stillness", "bond", "pact", "covenant",
    "alliance", "union", "loyalty", "devotion", "worship",
    # português
    "pacifismo", "fe", "coragem", "honra", "gloria", "destino", "fado",
    "juramento", "voto", "promessa", "maldicao", "bencao", "graca",
    "misericordia", "esperanca", "desespero", "medo", "ganancia",
    "raiva", "amor", "odio", "paz", "harmonia", "discordia", "ordem",
    "caos", "lei", "justica", "julgamento", "verdade", "mentira", "sonho",
    "memoria", "pensamento", "aura", "encantamento", "destino",
    "corrupcao", "pureza", "virtude", "pecado", "culpa", "tristeza",
    "alegria", "paixao", "desejo", "ambicao", "orgulho", "inveja",
    "silencio", "imobilidade", "vinculo", "pacto", "alianca", "uniao",
    "lealdade", "devocao", "adoracao",
}

ACOES: Set[str] = {
    # inglês
    "destroy", "exile", "murder", "slay", "kill", "burn", "sear",
    "scorch", "incinerate", "shock", "strike", "blast", "bolt", "counter",
    "counterspell", "negate", "cancel", "bounce", "return", "draw",
    "discard", "sacrifice", "tap", "untap", "damage", "heal", "gain",
    "lose", "search", "reveal", "scry", "surveil", "mill", "reanimate",
    "resurrect", "revive", "summon", "conjure", "create", "cast", "copy",
    "steal", "control", "pacify", "banish", "purge", "cleanse", "cure",
    "restore", "protect", "defend", "attack", "charge", "ambush",
    "growth", "recall", "counsel", "brainstorm", "anticipate",
    "assault", "raid", "invade", "conquer", "explore", "journey", "travel",
    "escape", "flee", "hide", "seek", "find", "discover", "learn",
    "study", "ponder", "think", "remember", "forget", "imagine", "wake",
    "sleep", "rise", "fall", "break", "shatter", "smash", "crush",
    "grind", "tear", "rip", "split", "merge", "bind", "unbind", "open",
    "close", "seal", "unseal", "lock", "unlock", "ignite", "freeze",
    "melt", "drown", "flood", "quake", "erupt", "explode", "implode",
    "collapse", "grow", "shrink", "transform", "change", "swap",
    "exchange", "trade", "give", "take", "offer", "demand", "command",
    "order", "call", "answer", "speak", "whisper", "shout", "sing",
    "dance", "play", "work", "build", "craft", "forge", "mend", "repair",
    "drain", "devour", "consume", "feed", "starve", "poison", "infect",
    "plague",
    # português
    "destruir", "exilar", "matar", "queimar", "incinerar", "chocar",
    "golpear", "explodir", "anular", "cancelar", "devolver", "comprar",
    "descartar", "sacrificar", "virar", "desvirar", "danificar", "curar",
    "ganhar", "perder", "buscar", "revelar", "videncia", "mill",
    "reanimar", "ressuscitar", "reviver", "invocar", "conjurar", "criar",
    "copiar", "roubar", "controlar", "pacificar", "banir", "purificar",
    "limpar", "restaurar", "proteger", "defender", "atacar", "carregar",
    "emboscar", "assaltar", "invadir", "conquistar", "explorar", "viajar",
    "escapar", "fugir", "esconder", "procurar", "encontrar", "descobrir",
    "aprender", "estudar", "ponderar", "pensar", "lembrar", "esquecer",
    "sonhar", "imaginar", "acordar", "dormir", "subir", "cair", "quebrar",
    "estilhacar", "esmagar", "rasgar", "dividir", "fundir", "amarrar",
    "soltar", "abrir", "fechar", "selar", "trancar", "destrancar",
    "acender", "congelar", "derreter", "afogar", "inundar", "tremer",
    "colapsar", "crescer", "encolher", "transformar", "mudar", "trocar",
    "dar", "tomar", "oferecer", "exigir", "comandar", "ordenar",
    "chamar", "responder", "falar", "sussurrar", "gritar", "cantar",
    "dancar", "jogar", "trabalhar", "construir", "forjar", "consertar",
    "desperdicar", "drenar", "devorar", "consumir", "alimentar",
    "envenenar", "infectar", "praguejar",
}

# Normaliza todas as listas uma vez (sem acento, minúsculas).
_LUGARES = {_normalizar(p) for p in LUGARES}
_CRIATURAS = {_normalizar(p) for p in CRIATURAS}
_OBJETOS = {_normalizar(p) for p in OBJETOS}
_ENCANTAMENTOS = {_normalizar(p) for p in ENCANTAMENTOS}
_ACOES = {_normalizar(p) for p in ACOES}

# Mapeamento de lexnames do WordNet -> tipo interno. O segundo elemento é
# a "inclinação" entre instantânea e feitiço, quando o tipo é ação:
# verbos e fenômenos puxam instantânea; substantivos de ação/evento/
# processo/tempo puxam feitiço.
MAPA_LEXNAME: Dict[str, Tuple[str, Optional[str]]] = {
    "noun.location": ("land", None),
    "noun.person": ("creature", None),
    "noun.animal": ("creature", None),
    "noun.plant": ("creature", None),
    "noun.artifact": ("artifact", None),
    "noun.object": ("artifact", None),
    "noun.substance": ("artifact", None),
    "noun.food": ("artifact", None),
    "noun.act": ("instant_sorcery", "sorcery"),
    "noun.event": ("instant_sorcery", "sorcery"),
    "noun.process": ("instant_sorcery", "sorcery"),
    "noun.time": ("instant_sorcery", "sorcery"),
    "noun.phenomenon": ("instant_sorcery", "instant"),
    "noun.attribute": ("enchantment", None),
    "noun.state": ("enchantment", None),
    "noun.feeling": ("enchantment", None),
    "noun.cognition": ("enchantment", None),
    "noun.communication": ("enchantment", None),
    "noun.motive": ("enchantment", None),
}


@dataclass(frozen=True)
class InferenciaTipo:
    """Resultado da inferência: tipo interno (ou None) + explicação."""

    tipo: Optional[str]
    motivo: str


# -- Utilidades linguísticas -----------------------------------------

def _tokens_originais(nome: str) -> List[str]:
    """Extrai palavras (só letras, preservando acentos e capitalização)."""
    return re.findall(r"[^\W\d_]+", str(nome), flags=re.UNICODE)


def _tokens(nome: str) -> List[str]:
    """Extrai palavras (só letras, preservando acentos) em minúsculas."""
    return [token.casefold() for token in _tokens_originais(nome)]


def _pos_tags(tokens: List[str]) -> List[str]:
    """POS tags do NLTK; se o tagger não estiver disponível, devolve NN
    para tudo (a inferência ainda funciona via WordNet/listas)."""
    if not tokens:
        return []
    try:
        _garantir_recursos()
        return [tag for _, tag in pos_tag(tokens)]
    except (LookupError, OSError):
        return ["NN"] * len(tokens)


def _lemas(token: str) -> Set[str]:
    """Formas bases do token segundo o WordNet (plural, conjugação etc.)."""
    lemas = {token}
    for pos in (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV):
        try:
            forma = wn.morphy(token, pos)
        except LookupError:
            forma = None
        if forma:
            lemas.add(forma)
    return lemas


def _formas_portugues(token: str) -> Set[str]:
    """Singularização ingênua para as listas em português."""
    formas = {token}
    if token.endswith("s") and len(token) > 3:
        formas.add(token[:-1])
    if token.endswith("es") and len(token) > 4:
        formas.add(token[:-2])
    return formas


def _formas(token: str) -> Set[str]:
    return {
        _normalizar(forma)
        for forma in (_lemas(token) | _formas_portugues(token))
    }


def _synset_primario(token: str):
    """Primeiro synset do token (tentando as formas base se necessário)."""
    for forma in [token] + sorted(_lemas(token) - {token}):
        try:
            synsets = wn.synsets(forma)
        except LookupError:
            return None
        if synsets:
            return synsets[0]
    return None


def _tem_verbo(token: str) -> bool:
    for forma in _lemas(token):
        try:
            if wn.synsets(forma, pos=wn.VERB):
                return True
        except LookupError:
            return False
    return False


def _indice_cabeca(tokens: List[str]) -> Optional[int]:
    """Índice da palavra "principal" do nome: a última palavra significativa
    antes de uma preposição ("Sword of Fire and Ice" -> sword), ou a última
    palavra significativa do nome."""
    uteis = [i for i, token in enumerate(tokens) if token not in STOPWORDS]
    if not uteis:
        return None

    pos_preposicao = next(
        (i for i, token in enumerate(tokens) if token in PREPOSICOES), None
    )
    if pos_preposicao is not None:
        antes = [i for i in uteis if i < pos_preposicao]
        if antes:
            return antes[-1]

    return uteis[-1]


def _token_parece_nome(token: str) -> bool:
    """Forma "pronunciável" de nome próprio: tem vogal e não tem uma
    sequência enorme de consoantes (rejeita "asdfgh", aceita "Chandra")."""
    if len(token) < 3:
        return False
    if not re.search(r"[aeiouyáéíóúâêôãõ]", token):
        return False
    if re.search(r"[bcdfghjklmnpqrstvwxyzç]{5,}", token):
        return False
    return True


def _parece_nome_proprio(nome: str, tokens: List[str]) -> bool:
    """Fallback para nomes próprios inventados quando NENHUMA palavra deu
    voto: exige capitalizada no texto original, nenhum dígito e forma
    pronunciável -- assim "Chandra"/"Vizzerdrix" passam, mas "asdfgh" e
    "jasdhahs12123" não."""
    if re.search(r"\d", nome):
        return False
    if not re.search(r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ]", nome):
        return False

    return any(
        token not in STOPWORDS and _token_parece_nome(token)
        for token in tokens
    )


# -- Inferência -------------------------------------------------------

def inferir_tipo_por_nome(
    nome: str, rng: Optional[random.Random] = None
) -> InferenciaTipo:
    """Infere o tipo de carta a partir do nome.

    Retorna `InferenciaTipo(tipo, motivo)`. `tipo` é uma chave de
    `TIPOS_VALIDOS` (inclusive "land") ou None quando não há sinal
    suficiente ou há empate -- nesse caso o chamador deve sortear.
    `motivo` explica a decisão em português (útil para debug/relatório).
    """
    rng = rng or random.Random()
    tokens = _tokens(nome)
    tokens_originais = _tokens_originais(nome)
    tem_digito = bool(re.search(r"\d", nome))
    if not tokens:
        return InferenciaTipo(None, "nome sem palavras reconhecíveis")

    tags = _pos_tags(tokens)
    cabeca = _indice_cabeca(tokens)
    votos: Dict[str, float] = defaultdict(float)
    votos_instant = 0.0
    votos_sorcery = 0.0
    evidencias: List[str] = []

    def votar(tipo: str, peso: float, evidencia: str, inclinacao=None):
        nonlocal votos_instant, votos_sorcery
        votos[tipo] += peso
        if inclinacao == "instant":
            votos_instant += peso
        elif inclinacao == "sorcery":
            votos_sorcery += peso
        if evidencia not in evidencias:
            evidencias.append(evidencia)

    for indice, token in enumerate(tokens):
        if token in STOPWORDS:
            continue

        formas = _formas(token)
        no_cabeca = indice == cabeca

        # 1) Listas curadas (têm prioridade; o WordNet também vota, mas as
        #    listas cobrem jargão de Magic e sentidos enganosos).
        em_lugar = formas & _LUGARES
        em_criatura = formas & _CRIATURAS
        em_objeto = formas & _OBJETOS
        em_encantamento = formas & _ENCANTAMENTOS
        em_acao = formas & _ACOES
        casou_curada = bool(
            em_lugar or em_criatura or em_objeto or em_encantamento or em_acao
        )

        if em_lugar:
            votar("land", 4.0, f"{token}: lugar")
        if em_criatura:
            votar("creature", 3.5 if no_cabeca else 1.5, f"{token}: profissão/criatura")
        if em_objeto:
            votar("artifact", 3.0 if no_cabeca else 1.5, f"{token}: objeto")
        if em_encantamento:
            votar(
                "enchantment",
                3.0 if no_cabeca else 1.5,
                f"{token}: conceito abstrato",
            )
        if em_acao:
            votar("instant_sorcery", 3.5 if no_cabeca else 2.0, f"{token}: ação")

        # 2) POS tagging: verbo em posição inicial é sinal forte de ação.
        tag = tags[indice] if indice < len(tags) else "NN"
        if tag in {"VB", "VBP", "VBZ", "VBD"} and _tem_verbo(token):
            votar(
                "instant_sorcery",
                4.0 if indice == 0 else 1.5,
                f"{token}: verbo",
                inclinacao="instant",
            )

        # 3) WordNet: categoria semântica do primeiro synset.
        synset = _synset_primario(token)
        if synset is not None:
            peso = 2.5 if no_cabeca else 1.0
            if synset.pos() == "v":
                votar(
                    "instant_sorcery",
                    peso,
                    f"{token}: verbo (WordNet)",
                    inclinacao="instant",
                )
            else:
                categoria = MAPA_LEXNAME.get(synset.lexname())
                if categoria is not None:
                    tipo, inclinacao = categoria
                    votar(
                        tipo,
                        peso,
                        f"{token}: {synset.lexname()}",
                        inclinacao=inclinacao,
                    )

        # 4) Nome próprio: a PRIMEIRA palavra capitalizada desconhecida do
        #    WordNet e das listas (ex: "Nicol", "Jace", "Nissa") puxa
        #    criatura -- mesmo que outra palavra do nome seja um substantivo
        #    comum (ex: "Nicol Bolas", em que "bolas" tem synset de
        #    artefato). Só a primeira palavra entra: em nomes como "Espada
        #    Runica", "Runica" é um adjetivo/nome de fantasia, não o núcleo
        #    do sintagma.
        if (
            indice == 0
            and not tem_digito
            and synset is None
            and not casou_curada
            and tokens_originais
            and tokens_originais[0][:1].isupper()
            and _token_parece_nome(token)
        ):
            votar("creature", 3.0, f"{token}: nome próprio")

    if not votos:
        if _parece_nome_proprio(nome, tokens):
            return InferenciaTipo(
                "creature",
                "nome próprio (sem sentido reconhecível no WordNet) -> criatura",
            )
        return InferenciaTipo(None, "sem sinal reconhecível -> tipo sorteado")

    ranking = sorted(votos.items(), key=lambda item: item[1], reverse=True)
    melhor_tipo, melhor_peso = ranking[0]

    if len(ranking) > 1 and abs(melhor_peso - ranking[1][1]) < 0.5:
        segundo_tipo, segundo_peso = ranking[1]
        nomes = {
            "instant_sorcery": "instantânea/feitiço",
            **{chave: valor.lower() for chave, valor in TIPOS_DISPLAY.items()},
        }
        return InferenciaTipo(
            None,
            f"empate entre {nomes.get(melhor_tipo, melhor_tipo)} e "
            f"{nomes.get(segundo_tipo, segundo_tipo)} -> tipo sorteado",
        )

    if melhor_peso < 1.5:
        return InferenciaTipo(None, "sinal fraco -> tipo sorteado")

    if melhor_tipo == "instant_sorcery":
        if votos_instant > votos_sorcery + 0.5:
            tipo_final = "instant"
        elif votos_sorcery > votos_instant + 0.5:
            tipo_final = "sorcery"
        else:
            tipo_final = rng.choice(["instant", "sorcery"])
        tipo_display = TIPOS_DISPLAY[tipo_final]
    else:
        tipo_final = melhor_tipo
        tipo_display = TIPOS_DISPLAY[tipo_final]

    detalhe = "; ".join(evidencias[:4])
    return InferenciaTipo(
        tipo_final,
        f"{tipo_display} (pistas: {detalhe})",
    )
