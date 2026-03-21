// arquivo init.c no mpmissions dayz server

void main()

{

    //INIT ECONOMY--------------------------------------

    Hive ce = CreateHive();

    if ( ce )

        ce.InitOffline();

    //DATE RESET AFTER ECONOMY INIT-------------------------

    int year, month, day, hour, minute;

    int reset_month = 9, reset_day = 20;

    GetGame().GetWorld().GetDate(year, month, day, hour, minute);
    if ((month == reset_month) && (day < reset_day))

    {

        GetGame().GetWorld().SetDate(year, reset_month, reset_day, hour, minute);

    }
    else
    {
        if ((month == reset_month + 1) && (day > reset_day))

        {

            GetGame().GetWorld().SetDate(year, reset_month, reset_day, hour, minute);

        }
        else
        {
            if ((month < reset_month) || (month > reset_month + 1))
            {

                GetGame().GetWorld().SetDate(year, reset_month, reset_day, hour, minute);

            }
        }
    }
}

class CustomMission: MissionServer
{
	float m_Timer = 0;
    void SetRandomHealth(EntityAI itemEnt)
    {
        if ( itemEnt )
        {
            float rndHlt = Math.RandomFloat( 0.45, 0.65 );
            itemEnt.SetHealth01( "", "", rndHlt );
        }
    }



    override PlayerBase CreateCharacter(PlayerIdentity identity, vector pos, ParamsReadContext ctx, string characterName)

    {

        Entity playerEnt;

        playerEnt = GetGame().CreatePlayer( identity, characterName, pos, 0, "NONE" );

        Class.CastTo( m_player, playerEnt );



        GetGame().SelectPlayer( identity, m_player );
        return m_player;

    }

    override void StartingEquipSetup(PlayerBase player, bool clothesChosen)
    {
        EntityAI itemClothing;
        EntityAI itemEnt;
        ItemBase itemBs;
        float rand;

        itemClothing = player.FindAttachmentBySlotName( "Body" );

        if ( itemClothing )

        {

            SetRandomHealth( itemClothing );
            itemEnt = itemClothing.GetInventory().CreateInInventory( "BandageDressing" );

            player.SetQuickBarEntityShortcut(itemEnt, 2);
            string chemlightArray[] = { "Chemlight_White", "Chemlight_Yellow", "Chemlight_Green", "Chemlight_Red" };
            int rndIndex = Math.RandomInt( 0, 4 );
            itemEnt = itemClothing.GetInventory().CreateInInventory( chemlightArray[rndIndex] );
            SetRandomHealth( itemEnt );
            player.SetQuickBarEntityShortcut(itemEnt, 1);

            rand = Math.RandomFloatInclusive( 0.0, 1.0 );

            if ( rand < 0.35 )
                itemEnt = player.GetInventory().CreateInInventory( "Apple" );
            else if ( rand > 0.65 )
                itemEnt = player.GetInventory().CreateInInventory( "Pear" );
            else
                itemEnt = player.GetInventory().CreateInInventory( "Plum" );

            player.SetQuickBarEntityShortcut(itemEnt, 3);

            SetRandomHealth( itemEnt );

        }

        itemClothing = player.FindAttachmentBySlotName( "Legs" );

        if ( itemClothing )
            SetRandomHealth( itemClothing );
        itemClothing = player.FindAttachmentBySlotName( "Feet" );
    }
	override void OnUpdate(float timeslice) {
        super.OnUpdate(timeslice);

        m_Timer += timeslice;
        
        if (m_Timer >= 1.0)
        {
            array<Man> players = new array<Man>;
            GetGame().GetPlayers(players);

            foreach (Man player : players)
            {
				PlayerBase pBody = PlayerBase.Cast(player);
                if (pBody) 
                {
                    vector pos = pBody.GetPosition();
					vector dir = pBody.GetDirection();

					int isAiming = 0;
					int isFiring = 0;
					string weaponName = "none";
					EntityAI itemInHand = pBody.GetHumanInventory().GetEntityInHands();
					if (itemInHand && itemInHand.IsWeapon()) {
						weaponName = itemInHand.GetType();
						if (pBody.GetInputController().IsWeaponRaised()) {
								isAiming = 1;
							}
					}
                    // DATA_LOG formatado para facilitar a leitura da IA depois
                    Print("DATA_LOG | " + GetGame().GetTime() + " | " + pBody.GetID() + " | " + pos[0] + " | " + pos[1] + " | " + pos[2] + " | " + dir[0] + " | " + dir[2] + " | " + isAiming + " | " + weaponName);
                }
            }
            m_Timer = 0;
        }
    }
};

Mission CreateCustomMission(string path)

{
    return new CustomMission();
}