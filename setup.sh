#!/bin/bash

# Obtain data
mkdir data
cd data
curl --header 'Host: doc-10-38-docs.googleusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:65.0) Gecko/20100101 Firefox/65.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' --header 'Accept-Language: es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3' --referer 'https://drive.google.com/a/pucp.pe/uc?id=1U55jRKHNqTplTPbF4lEjQMDKwp6V_my6&export=download' --cookie 'AUTH_0bl49mdmcbo80irb4rdjkgnre04gdie9_nonce=g016rkpnbdepq; _ga=GA1.2.1177697819.1525055967' --header 'Upgrade-Insecure-Requests: 1' 'https://doc-10-38-docs.googleusercontent.com/docs/securesc/moh3ovbqbojv2u94v8i2q0l1g916rgh0/10flju6bmmujpc1d6kvum93skk2btk2a/1552536000000/11122016764816395111/11122016764816395111/1U55jRKHNqTplTPbF4lEjQMDKwp6V_my6?e=download&h=14771753379018855219&nonce=g016rkpnbdepq&user=11122016764816395111&hash=fnolh1m9jlvg9qoicbgpti4b2o1g90l1' --output 'disambiguation_light.zip'
curl --header 'Host: doc-0k-38-docs.googleusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:65.0) Gecko/20100101 Firefox/65.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' --header 'Accept-Language: es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3' --referer 'https://drive.google.com/' --cookie 'AUTH_0bl49mdmcbo80irb4rdjkgnre04gdie9=11122016764816395111|1552536000000|76h3citm7hjrcga3slqekme9lr2mo6la; _ga=GA1.2.1177697819.1525055967' --header 'Upgrade-Insecure-Requests: 1' 'https://doc-0k-38-docs.googleusercontent.com/docs/securesc/moh3ovbqbojv2u94v8i2q0l1g916rgh0/nb5lll3d53olku7cd2thbjhvnfbhkkbe/1552536000000/11122016764816395111/11122016764816395111/1zWVKWIp-uioc6l_ZoTxuQ96KWMFeWjm5?h=14771753379018855219&e=download' --output 'translation.zip'
curl --header 'Host: doc-00-38-docs.googleusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:65.0) Gecko/20100101 Firefox/65.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' --header 'Accept-Language: es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3' --referer 'https://drive.google.com/drive/u/1/folders/1hyoJKzVvVrOKdcojPQAVZY_xm9rw1fBu' --cookie 'AUTH_0bl49mdmcbo80irb4rdjkgnre04gdie9=11122016764816395111|1552536000000|76h3citm7hjrcga3slqekme9lr2mo6la; _ga=GA1.2.1177697819.1525055967' --header 'Upgrade-Insecure-Requests: 1' 'https://doc-00-38-docs.googleusercontent.com/docs/securesc/moh3ovbqbojv2u94v8i2q0l1g916rgh0/cotmbqtcvn3alp74jfgu0hfr1mcugk16/1552536000000/11122016764816395111/11122016764816395111/1AImj2OZtiHWLn46gvxpXY56SMM6m02D4?h=14771753379018855219&e=download' --output 'serialize.zip'

curl --header 'Host: doc-0g-38-docs.googleusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:65.0) Gecko/20100101 Firefox/65.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' --header 'Accept-Language: es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3' --referer 'https://drive.google.com/' --cookie 'AUTH_0bl49mdmcbo80irb4rdjkgnre04gdie9=11122016764816395111|1552536000000|76h3citm7hjrcga3slqekme9lr2mo6la; _ga=GA1.2.1177697819.1525055967' --header 'Upgrade-Insecure-Requests: 1' 'https://doc-0g-38-docs.googleusercontent.com/docs/securesc/moh3ovbqbojv2u94v8i2q0l1g916rgh0/v0r07b6qboibljqlsu65k0u31apq83ji/1552536000000/11122016764816395111/11122016764816395111/1t8Yls9UnbfOlnx4yj-JIuP5XS8z_XYyx?h=14771753379018855219&e=download' --output 'environment.yml'

unzip serialize.zip
unzip disambiguation_light.zip
unzip translation.zip

rm serialize.zip
rm disambiguation_light.zip
rm translation.zip

# Activate environment
conda  env create --file=environment.yml
