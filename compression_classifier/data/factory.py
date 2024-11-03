from enum import Enum, auto

from compression_classifier.data.base import BaseDataLoader, StringDataLoader, Base64DataLoader

class EData(Enum):

    AG_NEWS = auto()
    DBPEDIA = auto()
    GENOMIC_DROSOPHILA = auto()
    GENOMIC_HUMAN_COHN = auto()
    GENOMIC_HUMAN_REGULATORY = auto()
    GENOMIC_HUMAN_PROMOTERS = auto()
    GENOMIC_HUMAN_OCR = auto()
    GENOMIC_HUMAN_WORM = auto()
    GISAID_SARS_COV_2_5 = auto()
    GISAID_SARS_COV_2_10 = auto()
    GISAID_SARS_COV_2_25 = auto()
    GISAID_SARS_COV_2_50 = auto()
    GISAID_SARS_COV_2_100 = auto()
    GISAID_SARS_COV_2_250 = auto()
    GISAID_SARS_COV_2_1000 = auto()
    GISAID_SARS_COV_2 = auto()
    KIN_NEWS = auto()
    MICROSOFT_MALWARE = auto()
    REUTERS_EIGHT = auto()
    SOGOU_NEWS = auto()
    SWAHILI_NEWS = auto()
    TWENTY_NEWS = auto()
    YAHOO_ANSWERS = auto()

def data_factory(folder : str, e_data : EData) -> BaseDataLoader:
    match e_data:
        case EData.AG_NEWS:
            return StringDataLoader(f'{folder}ag_news/')
        case EData.DBPEDIA:
            return StringDataLoader(f'{folder}dbpedia/')
        case EData.GENOMIC_DROSOPHILA:
            return StringDataLoader(f'{folder}genomic_drosophila_enhancers_stark/')
        case EData.GENOMIC_HUMAN_COHN:
            return StringDataLoader(f'{folder}genomic_human_enhancers_cohn/')
        case EData.GENOMIC_HUMAN_REGULATORY:
            return StringDataLoader(f'{folder}genomic_human_ensembl_regulatory/')
        case EData.GENOMIC_HUMAN_PROMOTERS:
            return StringDataLoader(f'{folder}genomic_human_nontata_promoters/')
        case EData.GENOMIC_HUMAN_OCR:
            return StringDataLoader(f'{folder}genomic_human_ocr_ensembl/')
        case EData.GENOMIC_HUMAN_WORM:
            return StringDataLoader(f'{folder}genomic_human_worm/')
        case EData.GISAID_SARS_COV_2_5:
            return StringDataLoader(f'{folder}gisaid_sars_cov2/5/')
        case EData.GISAID_SARS_COV_2_10:
            return StringDataLoader(f'{folder}gisaid_sars_cov2/10/')
        case EData.GISAID_SARS_COV_2_25:
            return StringDataLoader(f'{folder}gisaid_sars_cov2/25/')
        case EData.GISAID_SARS_COV_2_50:
            return StringDataLoader(f'{folder}gisaid_sars_cov2/50/')
        case EData.GISAID_SARS_COV_2_100:
            return StringDataLoader(f'{folder}gisaid_sars_cov2/100/')
        case EData.GISAID_SARS_COV_2_250:
            return StringDataLoader(f'{folder}gisaid_sars_cov2/250/')
        case EData.GISAID_SARS_COV_2_1000:
            return StringDataLoader(f'{folder}gisaid_sars_cov2/1000/')
        case EData.GISAID_SARS_COV_2:
            return StringDataLoader(f'{folder}gisaid_sars_cov2/full/')
        case EData.KIN_NEWS:
            return StringDataLoader(f'{folder}kinnews/')
        case EData.MICROSOFT_MALWARE:
            return Base64DataLoader(f'{folder}microsoft_malware/')
        case EData.REUTERS_EIGHT:
            return StringDataLoader(f'{folder}reuters_eight/')
        case EData.SOGOU_NEWS:
            return StringDataLoader(f'{folder}sogou_news/')
        case EData.SWAHILI_NEWS:
            return StringDataLoader(f'{folder}swahili_news/')
        case EData.TWENTY_NEWS:
            return StringDataLoader(f'{folder}twenty_news/')
        case EData.YAHOO_ANSWERS:
            return StringDataLoader(f'{folder}yahoo_answers/')
    raise ValueError('Unknown Data!')
