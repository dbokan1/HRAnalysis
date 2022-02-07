#analiza kolona iz eda:
    #city: 0.04 korelacija, velik broj gradova, nekoliko gradova sa vecim brojem-relativno nebalansiran, previse informacija za dobrog modela
    #city_development_index: 0.32 najjaca korelacija, dvije kategorije jako zastupljene (5k i 2.5k), trebalo bi balansirati i zadrzati
    #gender: slaba korelacija, disbalansiran, trebalo bi ukloniti
    #relevent_experience: 0.13 dobra korelacija, relativno balansirana (13k:6k- dvije kategorije)
    #enrolled_university: 0.15 dobra korelacija, relativno balansiran (13k:4k:1k), eventualno neke manje elemente duplirati
    #education_level: 0.08 relativno slaba korelacija (ujedno najvise koreliran sa target), blaze disbalansiran (11:4:2:1), nije ni sa cim posebno korelirana
    #major_discipline: 0.013 relativno slaba korelacija, tesko disbalansirana, trebalo bi ukloniti
    #experience: 0.005 slaba korelacija (najveca sa last new job 0.06), balansirana kolona
    #company_size: dobra korelacija (0.19 druga najveca), balansirana kolona
    #company_type: dobra korelacija (0.12 cetvrta najveca), disbalansirana kolona, trebalo bi balansirati i zadrzati
    #last_new_job: 0.01 slaba korelacija, relativno balansirana (jedna kolona je oko 40%, ostale uporedive), jaka korelacija sa relevent_experience, company_size i type
    #training_hours: 0.02 relativno slaba korelacija, balansiran set, slabo koreliran sa svime

#zadrzavamo: city_development_index, company_size, company_type(potrebno balansirati), relevent_experience, enrolled_university, last_new_job, city
#izbacujemo: gender, major_discipline, experience, education_level, training_hours


#kategoricki one hot koding:
# encoded_columns = pd.get_dummies(data['column'])
# data = data.join(encoded_columns).drop('column', axis=1)