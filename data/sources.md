
## Variables usadas como _baseline_ del análisis

| Nombre                          |   Fuente  |        Unidad        | Intervalo de datos | Descripción |
|---------------------------------|:---------:|:--------------------:|:------------------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `cc`                              |    INE    |           -          |          -         | Código de identificación única del municipio |
| `localizacion`                    |    INE    |           -          |          -         | Nombre del municipio |
| `superficie`                      |    INE    | Kilómetros cuadrados |          -         | Superficie del municipio |
| `altitud`                         |    INE    |        Metros        |          -         | Altitud del municipio |
| `geo_dens_poblacion`              |  ALMUDENA |   Nº Personas / Km2  |    [1985, 2024]    | Densidad de población |
| `geo_distancia_capital`           |  ALMUDENA |      Kilómetros      |          -         | Distancia en kilómetros a la capital de la comunidad |
| `n_viviendas_totales`             |  ALMUDENA |     Nº viviendas     |        2021        | Número de viviendas |
| `n_autobus`                       |  ALMUDENA |      Nº paradas      |    [1993, 2023]    | Número de paradas de autobús |
| `n_bibliotecas`                   |  ALMUDENA |    Nº bibliotecas    |    [1985, 2023]    | Número de bibliotecas |
| `n_centros_salud`                 |  ALMUDENA |      Nº centros      |    [2009, 2023]    | Número de consultorios locales y centros de salud |
| `n_centros_servsoc`               |  ALMUDENA |      Nº centros      |    [2014, 2024]    | Número de centros de servicios sociales |
| `n_cercanias`                     |  ALMUDENA |     Nº estaciones    |    [1993, 2023]    | Número de estaciones de cercanías |
| `n_cines`                         |  ALMUDENA |       Nº cines       |    [1985, 2023]    | Número de cines |
| `n_crecimiento_vegetativo`        |  ALMUDENA |      Nª Personas     |    [1985, 2022]    | Crecimiento vegetativo (nacimientos - defunciones) |
| `n_defunciones`                   |  ALMUDENA |      Nº personas     |    [1985, 2023]    | Número de defunciones de residentes |
| `n_farmacias`                     |  ALMUDENA |     Nº farmacias     |    [2009, 2023]    | Número de farmacias |
| `n_gasolineras`                   |  ALMUDENA |    Nº gasolineras    |  [2015, 2023] (**) | Número de gasolineras |
| `n_hoteles`                       |  ALMUDENA |  Nº establecimientos |    [2008, 2023]    | Número de establecimientos hoteleros (hoteles, viviendas turísticas, hostales, etc.) |
| `n_lineas_<XXXX>`                 |    CRTM   |       Nº líneas      |        2022        | Número de líneas de metro (<`metro`>), cercanías (<`cercanias`>), buses de la EMT (<`buses_EMT`>), autobuses urbanos (<`buses_urb`>) y autobuses interurbanos (<`buses_int`>) para cada municipio |
| `n_matrimonios`                   |  ALMUDENA |    Nº matrimonios    |    [2005, 2022]    | Número de matrimonios |
| `n_metro`                         |  ALMUDENA |     Nº estaciones    |    [2007, 2023]    | Número de estaciones de metro |
| `n_migr_inter`                    |  ALMUDENA |    Nº migraciones    |    [2021, 2023]    | Número de migraciones de residentes de la Comunidad de Madrid a otras CC.AA. |
| `n_migr_intra`                    |  ALMUDENA |    Nº migraciones    |    [2021, 2023]    | Número de migraciones de residentes de la Comunidad de Madrid a otros municipios de la Comunidad de Madrid |
| `n_nacimientos`                   |  ALMUDENA |      Nº personas     |    [1985, 2022]    | Número de nacimientos de madres residentes |
| `n_par_rusticas`                  |  ALMUDENA |      Nº parcelas     |    [1993, 2024]    | Número de parcelas rústicas |
| `n_par_urbanas_catastrales`       |  ALMUDENA |      Nº parcelas     |    [1994, 2024]    | Número de parcelas urbanas catastrales |
| `n_par_urbanas_sin_edificar`      |  ALMUDENA |      Nº parcelas     |    [1994, 2024]    | Número de parcelas urbanas sin edificar |
| `n_paro`                          |  ALMUDENA |      Nº personas     |    [2006, 2024]    | Número de parados |
| `n_plazas_hoteles`                |  ALMUDENA |       Nº plazas      |    [2008, 2023]    | Número de plazas en establecimientos hoteleros |
| `n_ptos_limpios`                  |  ALMUDENA |   Nº puntos limpios  |  [2013, 2023] (*)  | Número de puntos limpios |
| `n_ss_construccion`               |  ALMUDENA |      Nº personas     |    [2009, 2024]    | Número de afiliados dados de alta en la seguridad social en el sector construcción |
| `n_ss_dis_hos`                    |  ALMUDENA |      Nº personas     |    [2009, 2024]    | Número de afiliados dados de alta en la seguridad social en el sector de servicios de distribución y hostelería |
| `n_ss_emp_fin`                    |  ALMUDENA |      Nº personas     |    [2009, 2024]    | Número de afiliados dados de alta en la seguridad social en el sector empresas y financiero |
| `n_ss_general`                    |  ALMUDENA |      Nº personas     |    [2009, 2024]    | Número de afiliados dados de alta en la seguridad social |
| `n_ss_inmobiliarias`              |  ALMUDENA |      Nº personas     |    [2009, 2024]    | Número de afiliados dados de alta en la seguridad social en el sector inmobiliario |
| `n_ss_min_ind_ene`                |  ALMUDENA |      Nº personas     |    [2009, 2024]    | Número de afiliados dados de alta en la seguridad social en el sector minero, industrial y energético |
| `n_uu`                            |  ALMUDENA |  Nº unidades urbanas |    [1992, 2024]    | Número de unidades urbanas |
| `n_uu_deporte`                    |  ALMUDENA |  Nº unidades urbanas |    [2006, 2024]    | Número de unidades urbanas dedicadas a deporte |
| `n_uu_oficinas`                   |  ALMUDENA |  Nº unidades urbanas |    [2006, 2024]    | Número de unidades urbanas dedicadas a oficinas |
| `p_feminidad`                     |  ALMUDENA |         Ratio        |    [1996, 2024]    | Tasa de feminidad |
| `y_edad_media`                    |  ALMUDENA |         Años         |    [1998, 2024]    | Edad media |
| `n_tran_v2mano`                   |   BANVI   |   Nº transacciones   |    [2004, 2024]    | Número de transacciones inmobiliarias de viviendas de segunda mano |
| `n_tran_vlibre`                   |   BANVI   |   Nº transacciones   |    [2004, 2024]    | Número de transacciones inmobiliarias de viviendas libres |
| `n_tran_vnueva`                   |   BANVI   |   Nº transacciones   |    [2004, 2024]    | Número de transacciones inmobiliarias de viviendas nuevas |
| `n_tran_vprotegida`               |   BANVI   |   Nº transacciones   |    [2004, 2024]    | Número de transacciones inmobiliarias de viviendas protegidas |
| `eur_renta_b_xhab`                |    INE    |       Euros (€)      |    [2015, 2022]    | Renta bruta media por persona |
| `eur_renta_b_xhog`                |    INE    |       Euros (€)      |    [2015, 2022]    | Renta bruta media por hogar |
| `eur_base_urbana`                 |  ALMUDENA |       Euros (€)      |    [1990, 2023]    | Base imponible urbana por recibo |
| `eur_deuda_viva`                  |  ALMUDENA |       Euros (€)      |    [2008, 2023]    | Deuda viva |
| `eur_gastos`                      |  ALMUDENA |       Euros (€)      |    [2002, 2024]    | Gastos de los presupuestos iniciales |
| `eur_ingresos`                    |  ALMUDENA |       Euros (€)      |    [2002, 2024]    | Ingresos de los presupuestos iniciales |
| `eur_vcatastral_uu`               |  ALMUDENA |       Euros (€)      |    [1992, 2024]    | Valor catastral por unidad catastral |
| `list_ccaa_colindante`            |    CNIG   |           -          |          -         | Variable de la CC.AA. con la que colinda el municipio |
| `bool_colinda_otra_ccaa`          |    CNIG   |           -          |          -         | Variable dicotómica de si el municipio colinda con otra comunidad autónoma |
| (matriz) `bool_colindan_cc`       |    CNIG   |           -          |          -         | Variable dicotómica de si dos municipios colindan |
| (matriz) `n_grado_union_<XXXX>`   |    CRTM   |           -          |        2022        | Grado de unión en metro (<`metro`>), cercanías (<`cercanias`>), buses de la EMT (<`buses_EMT`>), autobuses urbanos (<`buses_urb`>) y autobuses interurbanos (<`buses_int`>) para los diferentes municipios que cuentan con transporte público |
| (matriz) `distancia_vuelo_pajaro` |    IECM   |      Kilómetros      |          -         | Distancia entre municipios a "vuelo de pájaro" |
| (matriz) `distancia_carretera`    |    IECM   |      Kilómetros      |          -         | Distancia entre municipios por carretera |
| (microdato) Viviendas en venta    | Idealista |           -          |        2022        | Anuncios de viviendas en venta en los diferentes municipios. Cuenta con las siguientes variables para cada vivienda en venta: `propertyCode` (Identificador único de la propiedad), `cc` (Código de identificación única del municipio), `price` (Precio total de venta del inmueble en euros), `propertyType` (Tipo de propiedad, e.g., flat, chalet, studio), `size` (Superficie construida en metros cuadrados), `rooms` (Número de habitaciones), `bathrooms` (Número de baños), `floor` (Planta en la que se encuentra la vivienda. Puede incluir valores como 'bj' para bajo), `exterior` (Indica si la propiedad es exterior (1) o interior (0)), `priceByArea` (Precio por metro cuadrado - €/m²), `status` (Estado del inmueble renovado, para reformar, etc. – puede estar vacía), `hasLift` (Indica si tiene ascensor), `hasAirConditioning` (Tiene aire acondicionado), `hasBoxRoom` (Tiene trastero), `hasGarden` (Tiene jardín), `hasSwimmingPool` (Tiene piscina), `hasTerrace` (Tiene terraza), `hasParkingSpace` (Dispone de plaza de aparcamiento), `hasStaging` (Decoración tipo 'home staging'), `has360` (Tiene tour en 360º disponible), `has3DTour` (Tiene tour 3D), `hasPlan` (Tiene plano de planta), `hasVideo` (Incluye vídeo en el anuncio), `municipality` (Nombre del municipio), `latitude` (Coordenada de latitud), `longitude` (Coordenada de longitud), `typology` (Clasificación del inmueble, Similar a propertyType), `newDevelopment` (Indica si es una promoción de obra nueva), `newDevelopmentFinished` (Indica si la obra nueva ya está terminada), `newProperty` (Si la propiedad es completamente nueva), `parkingSpacePrice` (Precio específico de la plaza de garaje, si no está incluida), `isParkingSpaceIncludedInPrice` (Indica si el precio del aparcamiento está incluido), `priceDropPercentage` (Porcentaje de bajada de precio), `priceDropValue` (Cantidad en euros que ha bajado el precio), `dropDate` (Fecha de la bajada de precio), `title` (Título del anuncio), `numPhotos` (Número de fotos del anuncio) y `description` (Descripción libre del anuncio). |

(*) Excepto 2015, 2016 y 2019
(**) Excepto 2016 y 2019


## Variables usadas en la validación global

| Nombre                          |   Fuente  |        Unidad        | Intervalo de datos | Descripción |
|---------------------------------|:---------:|:--------------------:|:------------------:| --------------------------- |
| `p_votos_PP_2023` | ALMUDENA | % personas | 2023 | Porcentaje de votos al PP en las elecciones al Congreso de los Diputados del año 2023 |
| `p_votos_PSOE_2023` | ALMUDENA | % personas | 2023 | Porcentaje de votos al PSOE en las elecciones al Congreso de los Diputados del año 2023 |
| `n_doc_cred_pre_hip` | ALMUDENA | Nº Documentos | 2022 | Número de documentos notariales de créditos, préstamos y garantías hipotecarias |
| `n_ongs` | ALMUDENA | Nº ONGs | 2022 | Número de documentos notariales de créditos, préstamos y garantías hipotecarias |
| `n_lin_tel_ATF` | ALMUDENA | Nº Líneas | 2022 | número de líneas telefónicas ATF |


## Fuentes de datos

* [IECM](https://gestiona.comunidad.madrid/iestadis/)
    * [ALMUDENA](https://gestiona.comunidad.madrid/desvan/Inicio.icm?enlace=almudena)
    * [BANVI](https://gestiona.comunidad.madrid/baco_web/html/web/AccionPaginaPrincipalVivienda.icm)
* [INE](https://www.ine.es/)
* [CNIG](https://centrodedescargas.cnig.es/CentroDescargas/home)
* [CRTM](https://datos.crtm.es/)
* [Idealista](https://www.idealista.com/data/)

