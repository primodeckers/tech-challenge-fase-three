"""Gera um CSV de laudos sintéticos para triagem (normal / atencao / urgente)."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

SEED = 42
N_POR_CLASSE = 1000
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "laudos.csv"

IDADES = list(range(18, 86))
SEXOS = ["masculino", "feminino"]

NORMAL = [
    "Paciente de {idade} anos, sexo {sexo}. Exame de rotina sem queixas. Achados dentro da normalidade. Sem sinais de alarme.",
    "Laudo de acompanhamento. Paciente {idade} anos ({sexo}). Exames laboratoriais estaveis. Sem alteracoes agudas.",
    "Check-up anual. {idade} anos, {sexo}. Hemograma e bioquimica sem desvios relevantes. Liberado para retorno em 12 meses.",
    "Radiografia de torax de controle. Paciente {idade} anos, {sexo}. Campos pulmonares limpos. Silhueta cardiaca normal.",
    "Consulta de revisao. {idade} anos, {sexo}. Paciente assintomatico. Pressao e frequencia cardiaca na linha de base.",
    "Resultado de ultrassom abdominal. {idade} anos, {sexo}. Figado, vesicula e rins sem alteracoes. Sem liquido livre.",
    "Laudo de eletrocardiograma de rotina. {idade} anos, {sexo}. Ritmo sinusal. Sem isquemia aguda.",
    "Retorno pos-tratamento. Paciente {idade} anos, {sexo}. Quadro resolvido. Sem necessidade de nova intervencao.",
    "Exame admissionais ocupacionais. {idade} anos, {sexo}. Apto. Sem achados patologicos.",
    "Endoscopia de acompanhamento. {idade} anos, {sexo}. Mucosa sem lesoes. Biopsias previas sem atipias.",
]

ATENCAO = [
    "Paciente de {idade} anos, sexo {sexo}. Febre ha 3 dias e mal-estar. Leucocitose leve. Requer observacao e reavaliacao em 24h.",
    "Laudo laboratorial. {idade} anos, {sexo}. Creatinina elevada em relacao ao basal. Monitorar funcao renal.",
    "Dor abdominal moderada. {idade} anos, {sexo}. Proteina C reativa aumentada. Sem sinais de peritonite. Investigar causa.",
    "Tosse persistente e febre baixa. {idade} anos, {sexo}. Radiografia com infiltrado inespecifico. Descartar infeccao respiratoria.",
    "Paciente {idade} anos, {sexo}. Glicemia de jejum alterada e sintomas leves. Ajuste de conduta e retorno breve.",
    "Cefaleia ha 48h sem deficit neurologico. {idade} anos, {sexo}. Analgesia sem resposta completa. Manter vigilância.",
    "Ultrassom mostra colecao pequena. {idade} anos, {sexo}. Sem criterios cirurgicos agora. Reavaliar se piora.",
    "Anemia nova em exame de rotina. {idade} anos, {sexo}. Investigar sangramento oculto. Encaminhar para acompanhamento.",
    "Pressao arterial descompensada em consulta. {idade} anos, {sexo}. Sem emergencia hipertensiva. Ajustar medicacao.",
    "Ferida operatoria com hiperemia leve. {idade} anos, {sexo}. Possivel inicio de infeccao. Revisar em 24 a 48h.",
]

URGENTE = [
    "Paciente de {idade} anos, sexo {sexo}. Dor toracica intensa e sudorese. ECG com supradesnivel. Suspeita de sindrome coronariana aguda.",
    "Laudo de emergencia. {idade} anos, {sexo}. Dispneia grave e saturacao baixa. Possivel embolia pulmonar. Encaminhar imediatamente.",
    "Rebaixamento do nivel de consciencia. {idade} anos, {sexo}. Sinais de AVC. Acionar protocolo de stroke.",
    "Abdome agudo. {idade} anos, {sexo}. Defesa peritoneal e leucocitose importante. Avaliacao cirurgica urgente.",
    "Sepse em investigacao. {idade} anos, {sexo}. Hipotensao, febre alta e lactato elevado. Iniciar protocolo sem atraso.",
    "Hemorragia digestiva com instabilidade. {idade} anos, {sexo}. Hemoglobina em queda. Risco imediato.",
    "Trauma craniano com vomitos e anisocoria. {idade} anos, {sexo}. TC imediata e suporte intensivo.",
    "Edema agudo de pulmao. {idade} anos, {sexo}. Ortopneia e crepitantes difusos. Internacao em unidade critica.",
    "Reacao anafilatica. {idade} anos, {sexo}. Broncoespasmo e hipotensao. Via aerea em risco.",
    "Cetoacidose. {idade} anos, {sexo}. Glicemia muito alta, vomitos e sonolencia. Reposicao e insulina agora.",
]


def _montar(templates: list[str], rng: random.Random) -> str:
    texto = rng.choice(templates).format(idade=rng.choice(IDADES), sexo=rng.choice(SEXOS))
    extras = [
        " Medico assistente solicita classificacao de urgencia.",
        " Encaminhado pela porta de entrada do hospital.",
        " Documento gerado para triagem automatica do laudo.",
        "",
    ]
    return texto + rng.choice(extras)


def gerar(n_por_classe: int = N_POR_CLASSE, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    linhas: list[dict[str, str]] = []
    for label, templates in (
        ("normal", NORMAL),
        ("atencao", ATENCAO),
        ("urgente", URGENTE),
    ):
        for _ in range(n_por_classe):
            linhas.append({"texto": _montar(templates, rng), "urgencia": label})
    rng.shuffle(linhas)
    return pd.DataFrame(linhas)


def main() -> None:
    df = gerar()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"arquivo: {OUT_PATH}")
    print(df["urgencia"].value_counts().to_string())


if __name__ == "__main__":
    main()
