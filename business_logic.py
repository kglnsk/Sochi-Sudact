from pydantic import BaseModel, Field

class Proxy(BaseModel):
    principal_name: str = Field(description="Имя доверителя")
    agent_name: str = Field(description="Имя поверенного")
    issue_date: str = Field(description="Дата выдачи")
    validity_period: str = Field(description="Срок действия")
    powers: str = Field(description="Полномочия")

class Contract(BaseModel):
    parties: str = Field(description="Стороны договора")
    subject: str = Field(description="Предмет договора")
    terms_of_performance: str = Field(description="Сроки исполнения")
    payment_terms: str = Field(description="Условия оплаты")
    liability: str = Field(description="Ответственность сторон")

class Act(BaseModel):
    act_number: str = Field(description="Номер акта")
    date_of_creation: str = Field(description="Дата составления")
    parties: str = Field(description="Стороны")
    list_of_works_services: str = Field(description="Перечень выполненных работ или услуг")
    signatures: str = Field(description="Подписи сторон")

class Application(BaseModel):
    applicant_name: str = Field(description="Имя заявителя")
    date_of_submission: str = Field(description="Дата подачи")
    purpose_of_application: str = Field(description="Цель заявления")
    applicant_signature: str = Field(description="Подпись заявителя")

class Order(BaseModel):
    order_number: str = Field(description="Номер приказа")
    date_of_issue: str = Field(description="Дата издания")
    content_of_order: str = Field(description="Содержание приказа")
    director_signature: str = Field(description="Подпись руководителя")

class Invoice(BaseModel):
    invoice_number: str = Field(description="Номер счета")
    date_of_issue: str = Field(description="Дата выставления")
    supplier: str = Field(description="Поставщик")
    buyer: str = Field(description="Покупатель")
    total_amount: str = Field(description="Сумма к оплате")

class Bill(BaseModel):
    appendix_number: str = Field(description="Номер приложения")
    date_of_creation: str = Field(description="Дата составления")
    description_of_documents: str = Field(description="Описание прилагаемых документов")

class Arrangement(BaseModel):
    parties: str = Field(description="Стороны соглашения")
    date_of_agreement: str = Field(description="Дата соглашения")
    subject_of_agreement: str = Field(description="Предмет соглашения")
    terms: str = Field(description="Условия")

class ContractOffer(BaseModel):
    subject_of_offer: str = Field(description="Предмет оферты")
    terms_of_offer: str = Field(description="Условия оферты")
    validity_period: str = Field(description="Срок действия")
    acceptance_procedure: str = Field(description="Порядок акцепта")

class Statute(BaseModel):
    organization_name: str = Field(description="Название организации")
    legal_address: str = Field(description="Юридический адрес")
    objectives: str = Field(description="Цели и задачи организации")
    rights_and_obligations: str = Field(description="Права и обязанности членов")

class Determination(BaseModel):
    decision_number: str = Field(description="Номер решения")
    date_of_decision: str = Field(description="Дата принятия")
    content_of_decision: str = Field(description="Содержание решения")
    participant_signatures: str = Field(description="Подписи участников")